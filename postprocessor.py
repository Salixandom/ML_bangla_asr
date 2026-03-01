"""
BanglaBERT Post-Processor for Bangla ASR

Uses the BanglaBERT ELECTRA pair for two-stage ASR output correction:

Stage 1 — Discriminator (csebuetnlp/banglabert):
    Identifies which tokens in the ASR output look "wrong" or out-of-place
    in context. Returns a binary label per token (0=ok, 1=suspicious).

Stage 2 — Generator (csebuetnlp/banglabert_generator):
    For each token flagged as suspicious, masks it and uses MLM to predict
    a better replacement from the surrounding context.

Why this works for ASR:
    CTC decoding produces acoustically plausible but sometimes linguistically
    unlikely sequences. BanglaBERT was pretrained on 27.5GB of clean Bangla
    text, so it has strong priors about which word sequences make sense.
    Tokens that the discriminator flags as "replaced" often correspond to
    ASR substitution errors that the generator can fix contextually.

Install dependencies:
    pip install transformers torch
    pip install git+https://github.com/csebuetnlp/normalizer

Usage:
    from postprocessor import BanglaBERTPostProcessor

    pp = BanglaBERTPostProcessor()
    corrected = pp.correct("আমি বাংলায় গাান গাই।")
    # → "আমি বাংলায় গান গাই।"

    # Batch correction
    corrected_list = pp.correct_batch(["text1", "text2", ...])

    # Also usable standalone:
    python postprocessor.py --text "আমি বাংলায় গাান গাই।"
    python postprocessor.py --input predictions.txt --output corrected.txt
"""

import torch
from pathlib import Path
from typing import List, Optional, Union
from transformers import (
    AutoModelForPreTraining,
    AutoTokenizer,
    pipeline,
)


def _normalize(text: str) -> str:
    """
    Apply csebuetnlp normalizer if available, else return text as-is.
    The normalizer standardizes Bangla Unicode (e.g. hasanta, nukta variants).
    Install with: pip install git+https://github.com/csebuetnlp/normalizer
    """
    try:
        from normalizer import normalize
        return normalize(text)
    except ImportError:
        # Warn once and proceed — correction still works, just less accurate
        if not hasattr(_normalize, '_warned'):
            print(
                "Warning: csebuetnlp normalizer not installed. "
                "Install with: pip install git+https://github.com/csebuetnlp/normalizer\n"
                "Post-processing will still run but may be less accurate."
            )
            _normalize._warned = True
        return text


class BanglaBERTPostProcessor:
    """
    Two-stage BanglaBERT post-processor for ASR transcription correction.

    Stage 1: Discriminator flags suspicious tokens
    Stage 2: Generator replaces flagged tokens using MLM context

    Args:
        discriminator_model: HF model name for the ELECTRA discriminator
        generator_model:     HF model name for the ELECTRA generator (MLM)
        device:              'cuda', 'cpu', or None (auto-detect)
        discrimination_threshold: Probability threshold above which a token
                                  is considered "replaced/wrong" (0.0–1.0).
                                  Lower = more aggressive correction.
                                  Default 0.5 (model's natural decision boundary).
        max_corrections_per_sentence: Safety cap — never replace more than
                                      this fraction of tokens in one sentence.
                                      Prevents over-correction.
        top_k:               Top-K candidates the generator considers per mask.
        generator_batch_size: Batch size for generator inference.
    """

    DISCRIMINATOR_MODEL = "csebuetnlp/banglabert"
    GENERATOR_MODEL     = "csebuetnlp/banglabert_generator"

    def __init__(
        self,
        discriminator_model: str = None,
        generator_model: str = None,
        device: Optional[str] = None,
        discrimination_threshold: float = 0.5,
        max_corrections_per_sentence: float = 0.3,
        top_k: int = 5,
        generator_batch_size: int = 16,
    ):
        self.discrimination_threshold = discrimination_threshold
        self.max_corrections_per_sentence = max_corrections_per_sentence
        self.top_k = top_k
        self.generator_batch_size = generator_batch_size

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        disc_name = discriminator_model or self.DISCRIMINATOR_MODEL
        gen_name  = generator_model     or self.GENERATOR_MODEL

        print(f"Loading BanglaBERT discriminator: {disc_name}")
        self.disc_tokenizer = AutoTokenizer.from_pretrained(disc_name)
        self.disc_model = AutoModelForPreTraining.from_pretrained(disc_name)
        self.disc_model.eval()
        self.disc_model.to(self.device)

        print(f"Loading BanglaBERT generator: {gen_name}")
        self.fill_mask = pipeline(
            "fill-mask",
            model=gen_name,
            tokenizer=gen_name,
            device=0 if self.device == 'cuda' else -1,
            top_k=top_k,
        )

        print(f"BanglaBERT post-processor ready on {self.device}")

    @torch.no_grad()
    def _detect_bad_tokens(self, normalized_text: str, encoding) -> tuple:
        """
        Run the ELECTRA discriminator on a pre-normalized sentence.

        FIX: Accepts the already-computed encoding instead of re-tokenizing.
        Previously this method called _normalize() and tokenized independently
        from _get_word_to_token_mapping(), causing the two tokenizations to
        diverge (different truncation points) which shifted the token→word
        alignment and caused word_idx out of range on long sentences.

        Args:
            normalized_text: Already-normalized text
            encoding: The BatchEncoding from _get_word_to_token_mapping,
                      reused here to guarantee identical tokenization.

        Returns:
            (bad_indices, token_probs)
        """
        inputs = {
            k: v.to(self.device)
            for k, v in encoding.items()
            if k != 'offset_mapping'
        }

        logits = self.disc_model(**inputs).logits  # (1, seq_len)
        probs = torch.sigmoid(logits.squeeze(0))   # (seq_len,)

        # Skip [CLS] (index 0) and [SEP] (last index)
        token_probs = probs[1:-1].cpu().tolist()

        bad_indices = [
            i for i, p in enumerate(token_probs)
            if p >= self.discrimination_threshold
        ]

        return bad_indices, token_probs

    def _get_word_to_token_mapping(self, normalized_text: str):
        """
        Tokenize normalized_text once and build word→token mapping.

        FIX: Returns the encoding so _detect_bad_tokens can reuse it,
        avoiding double-tokenization which could diverge on long inputs.

        FIX: word_idx is bounds-checked against len(words) — on truncated
        sentences the tokenizer's word_ids() can return an index ≥ n_words
        (e.g. when truncation cuts mid-word), causing IndexError downstream.

        Returns:
            (words, word_to_tokens, encoding)
        """
        words = normalized_text.split()
        n_words = len(words)

        encoding = self.disc_tokenizer(
            normalized_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        word_ids = encoding.word_ids(batch_index=0)

        word_to_tokens = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            # FIX: skip word_idx values that exceed the actual word list length.
            # This happens when the tokenizer truncates a sentence mid-word —
            # word_ids() may still emit an index for the truncated partial word
            # that points beyond words[].
            if word_idx >= n_words:
                continue
            adjusted_token_idx = token_idx - 1   # subtract [CLS] offset
            if adjusted_token_idx < 0:
                continue
            word_to_tokens.setdefault(word_idx, []).append(adjusted_token_idx)

        return words, word_to_tokens, encoding

    def correct(self, text: str, return_details: bool = False) -> Union[str, dict]:
        """
        Correct a single ASR transcription using BanglaBERT.

        Args:
            text: Raw ASR output string
            return_details: If True, return dict with correction details

        Returns:
            Corrected string, or dict with 'corrected', 'original',
            'flagged_words', 'corrections' if return_details=True
        """
        if not text or not text.strip():
            return text if not return_details else {
                'corrected': text, 'original': text,
                'flagged_words': [], 'corrections': []
            }

        normalized = _normalize(text)
        if not normalized.strip():
            return text if not return_details else {
                'corrected': text, 'original': text,
                'flagged_words': [], 'corrections': []
            }

        # Tokenize ONCE — reuse encoding for both Stage 1 and Stage 2.
        # FIX: Previously _detect_bad_tokens() and _get_word_to_token_mapping()
        # each tokenized independently. On long sentences (>512 tokens) they
        # could truncate at different points, shifting the token→word alignment
        # and causing word_idx out of range in Stage 2.
        try:
            words_list, word_to_tokens, encoding = self._get_word_to_token_mapping(normalized)
        except Exception:
            return text

        if not words_list:
            return text

        # Stage 1: Detect bad token indices (reuses encoding from above)
        bad_token_indices, token_probs = self._detect_bad_tokens(normalized, encoding)

        if not bad_token_indices:
            return text if not return_details else {
                'corrected': text, 'original': text,
                'flagged_words': [], 'corrections': []
            }

        # Find which words have at least one bad token
        bad_word_indices = set()
        for word_idx, token_indices in word_to_tokens.items():
            if any(t in bad_token_indices for t in token_indices):
                bad_word_indices.add(word_idx)

        # Safety cap: don't correct more than max_corrections_per_sentence fraction
        max_corrections = max(1, int(len(words_list) * self.max_corrections_per_sentence))
        if len(bad_word_indices) > max_corrections:
            # Keep only the most confident bad words (highest discriminator prob)
            word_scores = {}
            for word_idx in bad_word_indices:
                token_indices = word_to_tokens.get(word_idx, [])
                score = max(token_probs[t] for t in token_indices if t < len(token_probs))
                word_scores[word_idx] = score
            bad_word_indices = set(
                sorted(bad_word_indices, key=lambda i: word_scores[i], reverse=True)[:max_corrections]
            )

        if not bad_word_indices:
            return text if not return_details else {
                'corrected': text, 'original': text,
                'flagged_words': [], 'corrections': []
            }

        # Stage 2: Replace each bad word using the generator (MLM)
        corrected_words = words_list.copy()
        corrections = []
        mask_token = self.fill_mask.tokenizer.mask_token

        for word_idx in sorted(bad_word_indices):
            # Build masked sentence with this word replaced by [MASK]
            temp_words = corrected_words.copy()
            original_word = temp_words[word_idx]
            temp_words[word_idx] = mask_token
            masked_sentence = ' '.join(temp_words)

            try:
                results = self.fill_mask(masked_sentence)
                if results:
                    # Pick the top prediction
                    best = results[0]['token_str'].strip()
                    corrected_words[word_idx] = best
                    if best != original_word:
                        corrections.append({
                            'word_idx': word_idx,
                            'original': original_word,
                            'corrected': best,
                            'score': results[0]['score'],
                        })
            except Exception:
                # If generator fails for this word, leave it as-is
                pass

        corrected_text = ' '.join(corrected_words)

        if return_details:
            return {
                'corrected': corrected_text,
                'original': text,
                'flagged_words': [words_list[i] for i in sorted(bad_word_indices)],
                'corrections': corrections,
            }

        return corrected_text

    def correct_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        return_details: bool = False,
    ) -> List[Union[str, dict]]:
        """
        Correct a batch of ASR transcriptions.

        Args:
            texts: List of raw ASR output strings
            show_progress: Show tqdm progress bar
            return_details: If True, each result is a dict with correction details

        Returns:
            List of corrected strings (or dicts if return_details=True)
        """
        results = []

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="BanglaBERT correction")
            except ImportError:
                pass

        for text in iterator:
            result = self.correct(text, return_details=return_details)
            results.append(result)

        return results


class PostProcessedCTCDecoder:
    """
    Drop-in replacement for CTCDecoder that adds BanglaBERT post-processing.

    Wraps the existing CTCDecoder and applies BanglaBERT correction after
    greedy decoding. Use this in evaluate() and inference.

    Args:
        base_decoder: Your existing CTCDecoder instance
        postprocessor: BanglaBERTPostProcessor instance (or None to create one)
        enabled: Toggle post-processing on/off without changing code
    """

    def __init__(
        self,
        base_decoder,
        postprocessor: Optional[BanglaBERTPostProcessor] = None,
        enabled: bool = True,
        **postprocessor_kwargs,
    ):
        self.base_decoder = base_decoder
        self.enabled = enabled

        if enabled:
            if postprocessor is not None:
                self.postprocessor = postprocessor
            else:
                self.postprocessor = BanglaBERTPostProcessor(**postprocessor_kwargs)
        else:
            self.postprocessor = None

    def decode_greedy(self, logits: 'torch.Tensor') -> List[str]:
        """Greedy decode then apply BanglaBERT correction."""
        # Step 1: CTC greedy decode (existing logic)
        raw_predictions = self.base_decoder.decode_greedy(logits)

        if not self.enabled or self.postprocessor is None:
            return raw_predictions

        # Step 2: BanglaBERT correction
        corrected = self.postprocessor.correct_batch(
            raw_predictions,
            show_progress=False,
        )

        return corrected

    def decode_beam(self, logits: 'torch.Tensor', beam_width: int = 100) -> List[str]:
        """Beam search decode then apply BanglaBERT correction."""
        raw_predictions = self.base_decoder.decode_beam(logits, beam_width=beam_width)

        if not self.enabled or self.postprocessor is None:
            return raw_predictions

        corrected = self.postprocessor.correct_batch(
            raw_predictions,
            show_progress=False,
        )

        return corrected


# =============================================================================
# Integration guide for train.py
# =============================================================================
#
# In train.py Trainer.__init__, replace:
#
#     self.decoder = CTCDecoder(vocabulary)
#
# With:
#
#     from postprocessor import PostProcessedCTCDecoder
#     base_decoder = CTCDecoder(vocabulary)
#     self.decoder = PostProcessedCTCDecoder(
#         base_decoder=base_decoder,
#         enabled=config.inference.use_banglabert_correction,
#         discrimination_threshold=0.5,
#         max_corrections_per_sentence=0.3,
#     )
#
# In config.py InferenceConfig, the flag already exists:
#     use_banglabert_correction: bool = True
#
# That's it — no other changes needed. The PostProcessedCTCDecoder has the
# same interface as CTCDecoder so evaluate() works without modification.
# =============================================================================


def main():
    """Standalone CLI for testing post-processing."""
    import argparse

    parser = argparse.ArgumentParser(description='BanglaBERT ASR Post-Processor')
    parser.add_argument('--text', type=str, default=None,
                        help='Single text to correct')
    parser.add_argument('--input', type=str, default=None,
                        help='Input file with one transcription per line')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for corrected transcriptions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Discrimination threshold (default: 0.5)')
    parser.add_argument('--max-corrections', type=float, default=0.3,
                        help='Max fraction of words to correct per sentence (default: 0.3)')
    parser.add_argument('--details', action='store_true',
                        help='Print correction details')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu (default: auto)')
    args = parser.parse_args()

    pp = BanglaBERTPostProcessor(
        device=args.device,
        discrimination_threshold=args.threshold,
        max_corrections_per_sentence=args.max_corrections,
    )

    if args.text:
        result = pp.correct(args.text, return_details=args.details)
        if args.details:
            print(f"\nOriginal:  {result['original']}")
            print(f"Corrected: {result['corrected']}")
            if result['corrections']:
                print("\nCorrections made:")
                for c in result['corrections']:
                    print(f"  [{c['word_idx']}] '{c['original']}' → '{c['corrected']}' (score={c['score']:.3f})")
            else:
                print("No corrections made.")
        else:
            print(result)

    elif args.input:
        input_path = Path(args.input)
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.rstrip('\n') for line in f]

        print(f"Correcting {len(texts)} transcriptions...")
        corrected = pp.correct_batch(texts, show_progress=True)

        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in corrected:
                    f.write(line + '\n')
            print(f"Saved to {output_path}")
        else:
            for original, fixed in zip(texts, corrected):
                if original != fixed:
                    print(f"  ORIG: {original}")
                    print(f"  CORR: {fixed}")
                    print()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()