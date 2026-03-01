"""
Inference Pipeline for Bangla ASR

Complete inference including:
1. Audio preprocessing
2. Model inference (CTC decoding)
3. BanglaBERT text post-processing
4. Batch evaluation on manifest CSV (WER, CER, per-sample comparison)

Supports both greedy and beam search decoding.
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


# Global feature extractor (cached)
_FEATURE_EXTRACTOR = None

def get_feature_extractor(sample_rate: int = 16000) -> SeamlessM4TFeatureExtractor:
    """Get or create the feature extractor for wav2vec-BERT 2.0."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0",
            sampling_rate=sample_rate
        )
    return _FEATURE_EXTRACTOR


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
    """Per-sample WER with insertion/deletion/substitution breakdown."""
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


class ASRInference:
    """
    Complete ASR inference pipeline.

    Usage:
        inference = ASRInference.from_checkpoint('path/to/checkpoint')
        result    = inference.transcribe('audio.mp3')
        metrics   = inference.evaluate('processed/manifest.csv', split='valid')
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
        use_postprocessing: bool = True,
        device: str = "cuda"
    ) -> "ASRInference":
        checkpoint_dir = Path(checkpoint_dir)
        config = get_config()

        vocab = BanglaVocabulary(config.tokenizer)
        vocab_path = checkpoint_dir / 'vocabulary.json'
        if vocab_path.exists():
            vocab = BanglaVocabulary.load(vocab_path, config.tokenizer)

        # FIX: Disable freezing at inference time.
        # Wav2VecBertCTCModel.__init__ calls _freeze_feature_encoder() when
        # config.model.freeze_feature_encoder=True, which sets requires_grad=False
        # on half the encoder. Freezing is a training optimisation — at inference
        # all parameters are already inside torch.no_grad() so grad computation
        # never happens anyway. But the freeze printout ("Feature projection frozen",
        # "Frozen first 12/24 encoder layers") is misleading and the unnecessary
        # attribute writes slow down model loading slightly.
        config.model.freeze_feature_encoder = False

        model = Wav2VecBertCTCModel(
            config=config,
            vocab_size=len(vocab),
            pretrained_name=str(checkpoint_dir)
        )

        # During inference, never drop audio based on duration.
        # chunk_min_duration is a training-time filter to avoid noise clips
        # in the dataset. At inference time it would silently return an empty
        # transcription for any audio shorter than the threshold.
        # Setting it to 0.0 keeps every VAD segment regardless of length.
        config.audio.chunk_min_duration = 0.0

        preprocessor = AudioPreprocessor(config.audio, config.vad)
        decoder      = CTCDecoder(vocab)

        # FIX: from_checkpoint() creates a fresh config internally (defaults
        # use_banglabert_correction=False). The use_postprocessing argument is
        # the caller's intent — honour it directly instead of requiring the
        # config default to also be True.
        postprocessor = None
        if use_postprocessing:
            config.inference.use_banglabert_correction = True
        if use_postprocessing and config.inference.use_banglabert_correction:
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

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        return_chunks: bool = False,
        use_postprocessing: bool = True,
    ) -> Union[str, Dict]:
        processed = self.preprocessor.process_file(audio_path)
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
            input_features = features.input_features.to(self.device)

            outputs = self.model(input_features=input_features)
            logits  = outputs['logits']

            if self.config.decoding_method == 'beam':
                transcription = self.decoder.decode_beam(
                    logits, beam_width=self.config.beam_width
                )[0]
            else:
                transcription = self.decoder.decode_greedy(logits)[0]

            all_transcriptions.append(transcription)
            if return_chunks:
                chunk_results.append({'start': start_time, 'end': end_time, 'text': transcription})

        full_transcription = ' '.join(all_transcriptions)

        if use_postprocessing and self.postprocessor is not None:
            full_transcription = self.postprocessor.correct(full_transcription)
            if return_chunks:
                for chunk in chunk_results:
                    chunk['text_corrected'] = self.postprocessor.correct(chunk['text'])

        if return_chunks:
            return {'text': full_transcription, 'chunks': chunk_results,
                    'duration': processed.original_duration}
        return full_transcription

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        use_postprocessing: bool = True,
    ) -> List[str]:
        results = []
        for audio_path in tqdm(audio_paths, desc="Transcribing"):
            try:
                results.append(self.transcribe(audio_path, use_postprocessing=use_postprocessing))
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append("")
        return results

    def transcribe_dataset(
        self,
        audio_dir: Union[str, Path],
        output_csv: Union[str, Path],
        file_extension: str = '.mp3',
        use_postprocessing: bool = True,
    ) -> pd.DataFrame:
        audio_dir   = Path(audio_dir)
        audio_files = list(audio_dir.glob(f'*{file_extension}'))
        print(f"Found {len(audio_files)} audio files")

        results = []
        for audio_path in tqdm(audio_files, desc="Transcribing"):
            try:
                text = self.transcribe(audio_path, use_postprocessing=use_postprocessing)
                results.append({'id': audio_path.stem, 'sentence': text})
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({'id': audio_path.stem, 'sentence': ""})

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        return df

    # ------------------------------------------------------------------
    # Batch evaluation on manifest CSV
    # ------------------------------------------------------------------

    def evaluate(
        self,
        manifest_path: Union[str, Path],
        split: str = 'valid',
        output_csv: Optional[Union[str, Path]] = None,
        use_postprocessing: bool = True,
        num_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run batch evaluation on a manifest CSV and report WER / CER.

        The manifest CSV must have columns:
            audio_path  — path to chunk WAV file
            sentence    — reference transcription
            split       — 'train' / 'valid' / 'test'

        Args:
            manifest_path:     Path to processed/manifest.csv
            split:             Which split to evaluate
            output_csv:        Optional path to save per-sample CSV results
            use_postprocessing: Apply BanglaBERT correction before scoring
            num_samples:       Cap evaluation at N samples (None = all)
            verbose:           Print REF/PRED for every sample

        Returns:
            dict with wer, cer, error counts, and results_df (per-sample DataFrame)
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

        audio_col = 'audio_path' if 'audio_path' in df.columns else 'path'
        ref_col   = 'sentence'

        print(f"\nRunning inference on {len(df)} samples...")
        print(f"Post-processing: {'✅ BanglaBERT' if use_postprocessing and self.postprocessor else '❌ disabled'}")
        print(f"Decoding:        {self.config.decoding_method}")
        print()

        records   = []
        all_preds = []
        all_refs  = []

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
                    audio_path,
                    use_postprocessing=use_postprocessing,
                )
            except Exception as e:
                error_msg = str(e)
                tqdm.write(f"  ⚠️  Error on {sample_id}: {e}")

            details = compute_wer_details(prediction, reference)

            all_preds.append(prediction)
            all_refs.append(reference)

            total_ins       += details['insertions']
            total_del       += details['deletions']
            total_sub       += details['substitutions']
            total_ref_words += details['ref_words']
            total_hyp_words += details['hyp_words']

            records.append({
                'id':            sample_id,
                'audio_path':    str(audio_path),
                'reference':     reference,
                'prediction':    prediction,
                'wer':           round(details['wer'], 4),
                'insertions':    details['insertions'],
                'deletions':     details['deletions'],
                'substitutions': details['substitutions'],
                'ref_words':     details['ref_words'],
                'hyp_words':     details['hyp_words'],
                'error':         error_msg,
            })

            if verbose:
                icon = "✅" if details['wer'] == 0.0 else ("⚠️ " if details['wer'] < 0.3 else "❌")
                tqdm.write(f"  [{idx+1}/{len(df)}] {icon} WER={details['wer']:.3f}  id={sample_id}")
                tqdm.write(f"    REF:  {reference}")
                tqdm.write(f"    PRED: {prediction}")
                tqdm.write("")

        corpus_wer = compute_wer(all_preds, all_refs)
        corpus_cer = compute_cer(all_preds, all_refs)

        results_df = pd.DataFrame(records).sort_values('wer', ascending=False).reset_index(drop=True)

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n📄 Per-sample results saved to: {output_csv}")

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"EVALUATION RESULTS  —  split={split}  ({len(df)} samples)")
        print(f"{sep}")
        print(f"  WER:              {corpus_wer:.4f}  ({corpus_wer*100:.2f}%)")
        print(f"  CER:              {corpus_cer:.4f}  ({corpus_cer*100:.2f}%)")
        print(f"  Insertions:       {total_ins}")
        print(f"  Deletions:        {total_del}")
        print(f"  Substitutions:    {total_sub}")
        print(f"  Total ref words:  {total_ref_words}")
        print(f"  Total hyp words:  {total_hyp_words}")

        wer_vals = results_df['wer']
        print(f"\n  Per-sample WER distribution:")
        print(f"    Min:             {wer_vals.min():.4f}")
        print(f"    Median:          {wer_vals.median():.4f}")
        print(f"    Mean:            {wer_vals.mean():.4f}")
        print(f"    Max:             {wer_vals.max():.4f}")
        print(f"    Perfect (WER=0): {(wer_vals == 0).sum()} / {len(df)}")

        print(f"\n  Worst 10 samples:")
        for _, r in results_df.head(10).iterrows():
            print(f"    [{r['id']}]  WER={r['wer']:.3f}")
            print(f"      REF:  {r['reference']}")
            print(f"      PRED: {r['prediction']}")
        print(f"{sep}\n")

        return {
            'wer':                 corpus_wer,
            'cer':                 corpus_cer,
            'total_insertions':    total_ins,
            'total_deletions':     total_del,
            'total_substitutions': total_sub,
            'total_ref_words':     total_ref_words,
            'total_hyp_words':     total_hyp_words,
            'num_samples':         len(df),
            'results_df':          results_df,
        }


# -------------------------------------------------------------------------
# Competition submission
# -------------------------------------------------------------------------

def generate_submission(
    checkpoint_dir: Union[str, Path],
    test_audio_dir: Union[str, Path],
    sample_submission_path: Union[str, Path],
    output_path: Union[str, Path],
    use_postprocessing: bool = True,
    device: str = "cuda"
) -> pd.DataFrame:
    inference = ASRInference.from_checkpoint(
        checkpoint_dir, use_postprocessing=use_postprocessing, device=device
    )

    sample_df = pd.read_csv(sample_submission_path)
    test_ids  = sample_df['id'].tolist()
    test_dir  = Path(test_audio_dir)

    results = []
    for test_id in tqdm(test_ids, desc="Generating submission"):
        audio_path = None
        for ext in ['.mp3', '.wav', '.flac']:
            candidate = test_dir / f"{test_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            print(f"Warning: Audio not found for {test_id}")
            results.append({'id': test_id, 'sentence': ''})
            continue

        try:
            text = inference.transcribe(audio_path, use_postprocessing=use_postprocessing)
            results.append({'id': test_id, 'sentence': text})
        except Exception as e:
            print(f"Error processing {test_id}: {e}")
            results.append({'id': test_id, 'sentence': ''})

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return submission_df


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    import argparse

    parser     = argparse.ArgumentParser(description='Bangla ASR Inference & Evaluation')
    subparsers = parser.add_subparsers(dest='mode')

    # ---- evaluate ----
    ep = subparsers.add_parser('evaluate', help='Evaluate WER/CER on manifest split')
    ep.add_argument('--checkpoint',   required=True, help='Checkpoint directory')
    ep.add_argument('--manifest',     required=True, help='Path to processed/manifest.csv')
    ep.add_argument('--split',        default='valid',
                    help="Which split to evaluate: 'valid' (default), 'train', 'test'")
    ep.add_argument('--output',       default=None,
                    help='Save per-sample results to this CSV (sorted worst→best)')
    ep.add_argument('--num-samples',  type=int, default=None,
                    help='Limit to first N samples')
    ep.add_argument('--postprocess',  action='store_true',
                    help='Enable BanglaBERT post-processing')
    ep.add_argument('--beam-search',  action='store_true')
    ep.add_argument('--beam-width',   type=int, default=100)
    ep.add_argument('--device',       default='cuda')
    ep.add_argument('--no-verbose',   action='store_true',
                    help='Only print summary, suppress per-sample REF/PRED')

    # ---- transcribe ----
    tp = subparsers.add_parser('transcribe', help='Transcribe audio file(s)')
    tp.add_argument('--checkpoint',  required=True)
    tp.add_argument('--audio',       default=None, help='Single audio file')
    tp.add_argument('--audio-dir',   default=None, help='Directory of audio files')
    tp.add_argument('--output',      default='transcription.csv',
                      help='Output path. For --audio: saves a .txt file. '
                           'For --audio-dir: saves a .csv. '
                           'Default for --audio: <audio_stem>_transcription.txt')
    tp.add_argument('--postprocess', action='store_true')
    tp.add_argument('--beam-search', action='store_true')
    tp.add_argument('--beam-width',  type=int, default=100)
    tp.add_argument('--device',      default='cuda')

    # ---- submit ----
    sp = subparsers.add_parser('submit', help='Generate competition submission CSV')
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

    config = get_config()
    if getattr(args, 'beam_search', False):
        config.inference.decoding_method = 'beam'
        config.inference.beam_width      = args.beam_width
    if getattr(args, 'postprocess', False):
        config.inference.use_banglabert_correction = True

    if args.mode == 'evaluate':
        inference = ASRInference.from_checkpoint(
            args.checkpoint, use_postprocessing=args.postprocess, device=args.device
        )
        inference.evaluate(
            manifest_path=args.manifest,
            split=args.split,
            output_csv=args.output,
            use_postprocessing=args.postprocess,
            num_samples=args.num_samples,
            verbose=not args.no_verbose,
        )

    elif args.mode == 'transcribe':
        inference = ASRInference.from_checkpoint(
            args.checkpoint, use_postprocessing=args.postprocess, device=args.device
        )
        if args.audio:
            result = inference.transcribe(
                args.audio, return_chunks=True, use_postprocessing=args.postprocess
            )
            audio_stem = Path(args.audio).stem
            out_path   = Path(args.output) if args.output != 'transcription.csv'                          else Path(f"{audio_stem}_transcription.txt")

            # Write to txt — Windows terminal doesn't render Bangla Unicode
            lines = [
                f"File:     {args.audio}",
                f"Duration: {result['duration']:.2f}s",
                f"Chunks:   {len(result['chunks'])}",
                f"Decoding: {inference.config.decoding_method}",
                f"Post-processing: {'BanglaBERT' if inference.postprocessor else 'disabled'}",
                "",
                "--- TRANSCRIPTION ---",
                result['text'],
                "",
            ]
            if len(result['chunks']) > 1:
                lines.append("--- CHUNK BREAKDOWN ---")
                for i, chunk in enumerate(result['chunks']):
                    lines.append(f"[{chunk['start']:.2f}s - {chunk['end']:.2f}s]  {chunk['text']}")

            out_path.write_text("\n".join(lines), encoding='utf-8')
            print(f"\n✅ Transcription saved to: {out_path}")
            print(f"   Duration: {result['duration']:.2f}s  |  Chunks: {len(result['chunks'])}")
            # Print ASCII-safe summary; full Bangla text is in the file
            print(f"   (Open the .txt file to read the Bangla transcription)")
        elif args.audio_dir:
            inference.transcribe_dataset(
                audio_dir=args.audio_dir,
                output_csv=args.output,
                use_postprocessing=args.postprocess,
            )
        else:
            print("Please specify --audio or --audio-dir")

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