"""
Inference Pipeline for Bangla ASR

Complete inference including:
1. Audio preprocessing
2. Model inference (CTC decoding)
3. BanglaBERT text post-processing
4. Batch evaluation on manifest CSV (WER, CER, per-sample comparison)

Supports both greedy and beam search decoding.

Terminal output is kept ASCII-safe (no Bangla text printed).
All Bangla output goes to UTF-8 files that any text editor or Excel can open.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import soundfile as sf
from tqdm import tqdm
import pandas as pd

from transformers import SeamlessM4TFeatureExtractor

from config import PipelineConfig, InferenceConfig, get_config
from preprocessing import AudioPreprocessor
from dataset import BanglaVocabulary
from train import Wav2VecBertCTCModel, CTCDecoder
from postprocessor import BanglaBERTPostProcessor


# ---------------------------------------------------------------------------
# Feature extractor (cached globally — heavy to init)
# ---------------------------------------------------------------------------

_FEATURE_EXTRACTOR = None

def get_feature_extractor(sample_rate: int = 16000) -> SeamlessM4TFeatureExtractor:
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0",
            sampling_rate=sample_rate
        )
    return _FEATURE_EXTRACTOR


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_wer(predictions: list, references: list) -> float:
    import jiwer
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return 0.0
    return jiwer.wer([r for _, r in pairs], [p for p, _ in pairs])


def compute_cer(predictions: list, references: list) -> float:
    import jiwer
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return 0.0
    return jiwer.cer([r for _, r in pairs], [p for p, _ in pairs])


def compute_wer_details(prediction: str, reference: str) -> dict:
    """Per-sample WER with insertion / deletion / substitution counts."""
    import jiwer
    if not reference.strip():
        return {'wer': 0.0, 'insertions': 0, 'deletions': 0,
                'substitutions': 0, 'ref_words': 0, 'hyp_words': 0}
    out = jiwer.process_words(reference, prediction)
    return {
        'wer':           out.wer,
        'insertions':    out.insertions,
        'deletions':     out.deletions,
        'substitutions': out.substitutions,
        'ref_words':     len(reference.split()),
        'hyp_words':     len(prediction.split()),
    }


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class ASRInference:
    """
    Complete ASR inference pipeline.

    Typical usage:
        asr = ASRInference.from_checkpoint('output/best')
        asr.transcribe('audio.mp3')                     # → <stem>_transcription.txt
        asr.evaluate('processed/manifest.csv')           # → eval_valid_*.txt / .csv
    """

    def __init__(
        self,
        model: Wav2VecBertCTCModel,
        vocabulary: BanglaVocabulary,
        preprocessor: AudioPreprocessor,
        decoder: CTCDecoder,
        postprocessor: Optional[BanglaBERTPostProcessor] = None,
        config: Optional[InferenceConfig] = None,
        device: str = "cuda"
    ):
        self.model         = model
        self.vocabulary    = vocabulary
        self.preprocessor  = preprocessor
        self.decoder       = decoder
        self.postprocessor = postprocessor
        self.config        = config or InferenceConfig()
        self.device        = device

        self.model = self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        use_postprocessing: bool = False,
        device: str = "cuda"
    ) -> "ASRInference":
        """
        Load inference pipeline from a saved checkpoint directory.

        Args:
            checkpoint_dir:    Path to checkpoint (contains pytorch_model.bin,
                               vocabulary.json, etc.)
            use_postprocessing: Load and enable BanglaBERT 2-stage corrector.
                               Off by default — pass --postprocess in CLI to enable.
            device:            'cuda' or 'cpu'
        """
        checkpoint_dir = Path(checkpoint_dir)
        config = get_config()

        # Vocabulary
        vocab = BanglaVocabulary(config.tokenizer)
        vocab_path = checkpoint_dir / 'vocabulary.json'
        if vocab_path.exists():
            vocab = BanglaVocabulary.load(vocab_path, config.tokenizer)

        # Freezing is a training optimisation only. At inference all forward
        # passes are wrapped in torch.no_grad() — freezing is a no-op and only
        # prints misleading "Frozen first 12/24 encoder layers" messages.
        config.model.freeze_feature_encoder = False

        model = Wav2VecBertCTCModel(
            config=config,
            vocab_size=len(vocab),
            pretrained_name=str(checkpoint_dir)
        )

        # Never drop audio by duration at inference time. chunk_min_duration
        # is a training filter to discard noise clips. Leaving it non-zero
        # would silently return an empty string for any short clip.
        config.audio.chunk_min_duration = 0.0

        preprocessor = AudioPreprocessor(config.audio, config.vad)
        decoder      = CTCDecoder(vocab)

        # Honour use_postprocessing directly — from_checkpoint() creates a
        # fresh config where use_banglabert_correction=False by default, so
        # we must sync the flag before checking it.
        postprocessor = None
        if use_postprocessing:
            config.inference.use_banglabert_correction = True
            try:
                postprocessor = BanglaBERTPostProcessor(
                    discriminator_model=config.inference.banglabert_discriminator_model,
                    generator_model=config.inference.banglabert_generator_model,
                    device=device,
                    discrimination_threshold=config.inference.banglabert_discrimination_threshold,
                    max_corrections_per_sentence=config.inference.banglabert_max_corrections,
                )
                print("BanglaBERT post-processor loaded")
            except Exception as e:
                print(f"Warning: Could not initialize postprocessor: {e}")

        return cls(
            model=model, vocabulary=vocab, preprocessor=preprocessor,
            decoder=decoder, postprocessor=postprocessor,
            config=config.inference, device=device
        )

    # -----------------------------------------------------------------------
    # Core transcription
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        return_chunks: bool = False,
        use_postprocessing: bool = True,
    ) -> Union[str, Dict]:
        """
        Transcribe a single audio file.

        Args:
            audio_path:        Path to any audio format supported by librosa
            return_chunks:     If True, return dict with per-chunk timestamps
            use_postprocessing: Apply BanglaBERT correction (if loaded)

        Returns:
            Transcribed string, or dict {'text', 'chunks', 'duration'} when
            return_chunks=True
        """
        processed         = self.preprocessor.process_file(audio_path)
        feature_extractor = get_feature_extractor(processed.sample_rate)

        all_transcriptions = []
        chunk_results      = []

        for chunk_audio, start_time, end_time in processed.chunks:
            features = feature_extractor(
                chunk_audio,
                sampling_rate=processed.sample_rate,
                return_tensors="pt",
                padding=False
            )
            logits = self.model(
                input_features=features.input_features.to(self.device)
            )['logits']

            if self.config.decoding_method == 'beam':
                text = self.decoder.decode_beam(logits, beam_width=self.config.beam_width)[0]
            else:
                text = self.decoder.decode_greedy(logits)[0]

            all_transcriptions.append(text)
            if return_chunks:
                chunk_results.append({'start': start_time, 'end': end_time, 'text': text})

        full_text = ' '.join(all_transcriptions)

        if use_postprocessing and self.postprocessor is not None:
            full_text = self.postprocessor.correct(full_text)
            if return_chunks:
                for c in chunk_results:
                    c['text_corrected'] = self.postprocessor.correct(c['text'])

        if return_chunks:
            return {'text': full_text, 'chunks': chunk_results,
                    'duration': processed.original_duration}
        return full_text

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        use_postprocessing: bool = True,
    ) -> List[str]:
        """Transcribe a list of audio files, returning a list of strings."""
        results = []
        for p in tqdm(audio_paths, desc="Transcribing"):
            try:
                results.append(self.transcribe(p, use_postprocessing=use_postprocessing))
            except Exception as e:
                print(f"Error processing {p}: {e}")
                results.append("")
        return results

    def transcribe_dataset(
        self,
        audio_dir: Union[str, Path],
        output_csv: Union[str, Path],
        file_extension: str = '.mp3',
        use_postprocessing: bool = True,
    ) -> pd.DataFrame:
        """Transcribe all audio files in a directory and save to CSV."""
        audio_dir   = Path(audio_dir)
        audio_files = list(audio_dir.glob(f'*{file_extension}'))
        print(f"Found {len(audio_files)} audio files")

        records = []
        for p in tqdm(audio_files, desc="Transcribing"):
            try:
                text = self.transcribe(p, use_postprocessing=use_postprocessing)
                records.append({'id': p.stem, 'sentence': text})
            except Exception as e:
                print(f"Error processing {p}: {e}")
                records.append({'id': p.stem, 'sentence': ""})

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_csv}")
        return df

    # -----------------------------------------------------------------------
    # Batch evaluation on manifest CSV
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        manifest_path: Union[str, Path],
        split: str = 'valid',
        output_prefix: Optional[Union[str, Path]] = None,
        use_postprocessing: bool = True,
        num_samples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate WER / CER on a manifest CSV split and write results to files.

        Always writes three output files (all UTF-8 — open in any editor or Excel):
            <prefix>_summary.txt      — overall metrics, distribution, worst-10
            <prefix>_per_sample.txt   — every REF/PRED pair, sorted worst → best
            <prefix>_per_sample.csv   — full spreadsheet (utf-8-sig for Excel)

        The default prefix is  eval_<split>  in the current directory, so running:
            asr.evaluate('processed/manifest.csv')
        produces:
            eval_valid_summary.txt
            eval_valid_per_sample.txt
            eval_valid_per_sample.csv

        Args:
            manifest_path:     Path to processed/manifest.csv
            split:             'valid' (default), 'train', or 'test'
            output_prefix:     Base path for the three output files.
                               e.g. 'results/run1' →
                                   results/run1_summary.txt
                                   results/run1_per_sample.txt
                                   results/run1_per_sample.csv
            use_postprocessing: Apply BanglaBERT correction before scoring
            num_samples:       Evaluate only first N samples (None = all)

        Returns:
            dict with wer, cer, error counts, file paths, and results_df
        """
        manifest_path = Path(manifest_path)
        df = pd.read_csv(manifest_path)

        if 'split' in df.columns:
            df = df[df['split'] == split].reset_index(drop=True)
            print(f"Evaluating on '{split}' split: {len(df)} samples")
        else:
            print(f"No 'split' column — evaluating all {len(df)} samples")

        if len(df) == 0:
            print(f"No samples found for split='{split}'")
            return {}

        if num_samples is not None:
            df = df.head(num_samples)
            print(f"Limiting to first {num_samples} samples")

        audio_col  = 'audio_path' if 'audio_path' in df.columns else 'path'
        ref_col    = 'sentence'
        post_label = 'BanglaBERT' if use_postprocessing and self.postprocessor else 'disabled'

        # Resolve output file paths
        base = Path(output_prefix) if output_prefix else Path(f"eval_{split}")
        base.parent.mkdir(parents=True, exist_ok=True)
        summary_txt    = Path(str(base) + "_summary.txt")
        per_sample_txt = Path(str(base) + "_per_sample.txt")
        per_sample_csv = Path(str(base) + "_per_sample.csv")

        print(f"\nPost-processing : {post_label}")
        print(f"Decoding        : {self.config.decoding_method}")
        print(f"Output prefix   : {base}")
        print()

        # ---------------------------------------------------------------
        # Inference loop
        # ---------------------------------------------------------------
        records         = []
        all_preds       = []
        all_refs        = []
        total_ins = total_del = total_sub = 0
        total_ref_words = total_hyp_words = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            audio_path = Path(row[audio_col])
            reference  = str(row[ref_col]).strip() if pd.notna(row[ref_col]) else ""
            sample_id  = row.get('id', row.get('original_id', str(idx)))

            prediction = ""
            error_msg  = ""
            try:
                prediction = self.transcribe(
                    audio_path, use_postprocessing=use_postprocessing
                )
            except Exception as e:
                error_msg = str(e)
                tqdm.write(f"  ERROR [{sample_id}]: {e}")

            d = compute_wer_details(prediction, reference)
            all_preds.append(prediction)
            all_refs.append(reference)

            total_ins       += d['insertions']
            total_del       += d['deletions']
            total_sub       += d['substitutions']
            total_ref_words += d['ref_words']
            total_hyp_words += d['hyp_words']

            records.append({
                'id':            sample_id,
                'audio_path':    str(audio_path),
                'reference':     reference,
                'prediction':    prediction,
                'wer':           round(d['wer'], 4),
                'insertions':    d['insertions'],
                'deletions':     d['deletions'],
                'substitutions': d['substitutions'],
                'ref_words':     d['ref_words'],
                'hyp_words':     d['hyp_words'],
                'error':         error_msg,
            })

        corpus_wer = compute_wer(all_preds, all_refs)
        corpus_cer = compute_cer(all_preds, all_refs)

        results_df = (
            pd.DataFrame(records)
            .sort_values('wer', ascending=False)
            .reset_index(drop=True)
        )
        wer_vals = results_df['wer']

        # ---------------------------------------------------------------
        # Write per-sample CSV (utf-8-sig BOM lets Excel open Bangla correctly)
        # ---------------------------------------------------------------
        results_df.to_csv(per_sample_csv, index=False, encoding='utf-8-sig')

        # ---------------------------------------------------------------
        # Build summary block (shared between txt file and terminal print)
        # ---------------------------------------------------------------
        sep = "=" * 65
        summary_lines = [
            sep,
            f"EVALUATION SUMMARY  —  split={split}  ({len(df)} samples)",
            sep,
            f"  Decoding        : {self.config.decoding_method}",
            f"  Post-processing : {post_label}",
            "",
            f"  WER             : {corpus_wer:.4f}  ({corpus_wer*100:.2f}%)",
            f"  CER             : {corpus_cer:.4f}  ({corpus_cer*100:.2f}%)",
            "",
            f"  Insertions      : {total_ins}",
            f"  Deletions       : {total_del}",
            f"  Substitutions   : {total_sub}",
            f"  Total ref words : {total_ref_words}",
            f"  Total hyp words : {total_hyp_words}",
            "",
            f"  Per-sample WER distribution:",
            f"    Min             : {wer_vals.min():.4f}",
            f"    Median          : {wer_vals.median():.4f}",
            f"    Mean            : {wer_vals.mean():.4f}",
            f"    Max             : {wer_vals.max():.4f}",
            f"    Perfect (WER=0) : {(wer_vals == 0).sum()} / {len(df)}"
            f"  ({100*(wer_vals==0).sum()/len(df):.1f}%)",
            "",
            "  Worst 10 samples:",
        ]
        for _, r in results_df.head(10).iterrows():
            summary_lines.append(f"    [{r['id']}]  WER={r['wer']:.3f}")
            summary_lines.append(f"      REF:  {r['reference']}")
            summary_lines.append(f"      PRED: {r['prediction']}")
        summary_lines += [
            "",
            "  Output files:",
            f"    {summary_txt}",
            f"    {per_sample_txt}",
            f"    {per_sample_csv}",
            sep,
        ]

        # ---------------------------------------------------------------
        # Write summary txt
        # ---------------------------------------------------------------
        summary_txt.write_text("\n".join(summary_lines) + "\n", encoding='utf-8')

        # ---------------------------------------------------------------
        # Write per-sample detail txt  (sorted worst → best)
        # ---------------------------------------------------------------
        header = [
            sep,
            f"PER-SAMPLE DETAIL  —  split={split}  ({len(df)} samples)",
            "Sorted worst → best by WER",
            sep,
            "",
        ]
        detail_lines = []
        for rank, (_, r) in enumerate(results_df.iterrows(), 1):
            icon = "OK " if r['wer'] == 0.0 else ("WRN" if r['wer'] < 0.3 else "ERR")
            detail_lines.append(
                f"[{rank:>6}/{len(df)}] {icon} WER={r['wer']:.3f}  id={r['id']}\n"
                f"  REF:  {r['reference']}\n"
                f"  PRED: {r['prediction']}\n"
            )
        per_sample_txt.write_text(
            "\n".join(header + detail_lines), encoding='utf-8'
        )

        # ---------------------------------------------------------------
        # Terminal — metrics only (no Bangla text, safe for any terminal)
        # ---------------------------------------------------------------
        print(f"\n{sep}")
        print(f"EVALUATION COMPLETE  —  split={split}  ({len(df)} samples)")
        print(f"{sep}")
        print(f"  WER  : {corpus_wer:.4f}  ({corpus_wer*100:.2f}%)")
        print(f"  CER  : {corpus_cer:.4f}  ({corpus_cer*100:.2f}%)")
        print(f"  Ins  : {total_ins}   Del : {total_del}   Sub : {total_sub}")
        print(f"  Perfect (WER=0) : {(wer_vals==0).sum()} / {len(df)}")
        print()
        print("  Open these files to see full results in Bangla:")
        print(f"    Summary     -> {summary_txt}")
        print(f"    Per-sample  -> {per_sample_txt}")
        print(f"    Spreadsheet -> {per_sample_csv}")
        print(f"{sep}\n")

        return {
            'wer':                  corpus_wer,
            'cer':                  corpus_cer,
            'total_insertions':     total_ins,
            'total_deletions':      total_del,
            'total_substitutions':  total_sub,
            'total_ref_words':      total_ref_words,
            'total_hyp_words':      total_hyp_words,
            'num_samples':          len(df),
            'results_df':           results_df,
            'summary_txt':          summary_txt,
            'per_sample_txt':       per_sample_txt,
            'per_sample_csv':       per_sample_csv,
        }


# ---------------------------------------------------------------------------
# Competition submission
# ---------------------------------------------------------------------------

def generate_submission(
    checkpoint_dir: Union[str, Path],
    test_audio_dir: Union[str, Path],
    sample_submission_path: Union[str, Path],
    output_path: Union[str, Path],
    use_postprocessing: bool = False,
    device: str = "cuda"
) -> pd.DataFrame:
    """Generate competition submission CSV from test audio directory."""
    inference = ASRInference.from_checkpoint(
        checkpoint_dir, use_postprocessing=use_postprocessing, device=device
    )

    sample_df = pd.read_csv(sample_submission_path)
    test_dir  = Path(test_audio_dir)

    results = []
    for test_id in tqdm(sample_df['id'].tolist(), desc="Generating submission"):
        audio_path = None
        for ext in ['.mp3', '.wav', '.flac']:
            candidate = test_dir / f"{test_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            print(f"Warning: audio not found for {test_id}")
            results.append({'id': test_id, 'sentence': ''})
            continue

        try:
            text = inference.transcribe(audio_path, use_postprocessing=use_postprocessing)
            results.append({'id': test_id, 'sentence': text})
        except Exception as e:
            print(f"Error on {test_id}: {e}")
            results.append({'id': test_id, 'sentence': ''})

    submission_df = pd.DataFrame(results)
    # utf-8-sig so Excel opens the Bangla column correctly
    submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Submission saved to {output_path}")
    return submission_df


# ---------------------------------------------------------------------------
# CLI  (subcommands: evaluate | transcribe | submit)
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser     = argparse.ArgumentParser(
        description='Bangla ASR — Inference & Evaluation',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode')

    # ---- evaluate --------------------------------------------------------
    ep = subparsers.add_parser(
        'evaluate',
        help='Evaluate WER/CER on a manifest split, write results to files'
    )
    ep.add_argument('--checkpoint',    required=True,
                    help='Path to model checkpoint directory')
    ep.add_argument('--manifest',      required=True,
                    help='Path to processed/manifest.csv')
    ep.add_argument('--split',         default='valid',
                    help="Split to evaluate: 'valid' (default), 'train', 'test'")
    ep.add_argument('--output-prefix', default=None,
                    help=(
                        "Base path for output files.\n"
                        "e.g. --output-prefix results/run1  produces:\n"
                        "  results/run1_summary.txt\n"
                        "  results/run1_per_sample.txt\n"
                        "  results/run1_per_sample.csv\n"
                        "Default: eval_<split> in current directory."
                    ))
    ep.add_argument('--num-samples',   type=int, default=None,
                    help='Evaluate only first N samples')
    ep.add_argument('--postprocess',   action='store_true',
                    help='Enable BanglaBERT 2-stage correction')
    ep.add_argument('--beam-search',   action='store_true',
                    help='Use beam search decoding (default: greedy)')
    ep.add_argument('--beam-width',    type=int, default=100)
    ep.add_argument('--device',        default='cuda')

    # ---- transcribe ------------------------------------------------------
    tp = subparsers.add_parser(
        'transcribe',
        help='Transcribe one audio file or a directory of files'
    )
    tp.add_argument('--checkpoint',  required=True)
    tp.add_argument('--audio',       default=None,
                    help='Single audio file → writes <stem>_transcription.txt')
    tp.add_argument('--audio-dir',   default=None,
                    help='Directory of audio files → writes CSV')
    tp.add_argument('--output',      default=None,
                    help='Custom output path (optional)')
    tp.add_argument('--postprocess', action='store_true',
                    help='Enable BanglaBERT correction')
    tp.add_argument('--beam-search', action='store_true')
    tp.add_argument('--beam-width',  type=int, default=100)
    tp.add_argument('--device',      default='cuda')

    # ---- submit ----------------------------------------------------------
    sp = subparsers.add_parser(
        'submit',
        help='Generate competition submission CSV from test audio directory'
    )
    sp.add_argument('--checkpoint',        required=True)
    sp.add_argument('--test-dir',          required=True)
    sp.add_argument('--sample-submission', required=True)
    sp.add_argument('--output',            default='submission.csv')
    sp.add_argument('--postprocess',       action='store_true')
    sp.add_argument('--device',            default='cuda')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    # Shared config tweaks
    config = get_config()
    if getattr(args, 'beam_search', False):
        config.inference.decoding_method = 'beam'
        config.inference.beam_width      = args.beam_width

    # ---- evaluate --------------------------------------------------------
    if args.mode == 'evaluate':
        asr = ASRInference.from_checkpoint(
            args.checkpoint,
            use_postprocessing=args.postprocess,
            device=args.device,
        )
        asr.evaluate(
            manifest_path=args.manifest,
            split=args.split,
            output_prefix=args.output_prefix,
            use_postprocessing=args.postprocess,
            num_samples=args.num_samples,
        )

    # ---- transcribe ------------------------------------------------------
    elif args.mode == 'transcribe':
        asr = ASRInference.from_checkpoint(
            args.checkpoint,
            use_postprocessing=args.postprocess,
            device=args.device,
        )

        if args.audio:
            result   = asr.transcribe(
                args.audio, return_chunks=True,
                use_postprocessing=args.postprocess,
            )
            stem     = Path(args.audio).stem
            out_path = Path(args.output) if args.output \
                       else Path(f"{stem}_transcription.txt")

            lines = [
                f"File            : {args.audio}",
                f"Duration        : {result['duration']:.2f}s",
                f"Chunks          : {len(result['chunks'])}",
                f"Decoding        : {asr.config.decoding_method}",
                f"Post-processing : {'BanglaBERT' if asr.postprocessor else 'disabled'}",
                "",
                "--- TRANSCRIPTION ---",
                result['text'],
                "",
            ]
            if len(result['chunks']) > 1:
                lines.append("--- CHUNK BREAKDOWN ---")
                for c in result['chunks']:
                    lines.append(
                        f"[{c['start']:.2f}s - {c['end']:.2f}s]  {c['text']}"
                    )

            out_path.write_text("\n".join(lines), encoding='utf-8')
            print(f"\nTranscription saved to: {out_path}")
            print(f"Duration : {result['duration']:.2f}s  |  Chunks : {len(result['chunks'])}")
            print("(Open the .txt file to read the Bangla text)")

        elif args.audio_dir:
            out = Path(args.output) if args.output else Path("transcriptions.csv")
            asr.transcribe_dataset(
                audio_dir=args.audio_dir,
                output_csv=out,
                use_postprocessing=args.postprocess,
            )
        else:
            print("Please specify --audio or --audio-dir")

    # ---- submit ----------------------------------------------------------
    elif args.mode == 'submit':
        generate_submission(
            checkpoint_dir=args.checkpoint,
            test_audio_dir=args.test_dir,
            sample_submission_path=args.sample_submission,
            output_path=args.output,
            use_postprocessing=args.postprocess,
            device=args.device,
        )


if __name__ == "__main__":
    main()