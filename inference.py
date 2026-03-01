"""
Inference Pipeline for Bangla ASR

Complete inference including:
1. Audio preprocessing
2. Model inference (CTC decoding)
3. BanglaBERT text post-processing

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

# CHANGE 1: Import the real postprocessor instead of using the old inline class.
# The old BanglaBERTPostProcessor in this file only detected errors but never
# corrected them (banglabert_detection=False, banglat5_correction=False by default).
# The new one from postprocessor.py does full 2-stage correction:
#   Stage 1 - discriminator flags bad tokens
#   Stage 2 - generator (MLM) replaces them
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


# CHANGE 2: Removed the old BanglaBERTPostProcessor class entirely.
# It was a stub — use_banglabert_detection defaulted to False and
# use_banglat5_correction defaulted to False, so process() only ran
# clean_asr_output() and normalize_text() — no actual correction at all.
# The real implementation is now in postprocessor.py.


class ASRInference:
    """
    Complete ASR inference pipeline.

    Usage:
        inference = ASRInference.from_checkpoint('path/to/checkpoint')
        result = inference.transcribe('audio.mp3')
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
        self.model = model
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        self.decoder = decoder
        self.postprocessor = postprocessor
        self.config = config or InferenceConfig()
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        use_postprocessing: bool = True,
        device: str = "cuda"
    ) -> "ASRInference":
        """
        Load inference pipeline from checkpoint.

        Args:
            checkpoint_dir: Path to saved checkpoint
            use_postprocessing: Whether to use BanglaBERT post-processing
            device: Device to use

        Returns:
            ASRInference instance
        """
        checkpoint_dir = Path(checkpoint_dir)
        config = get_config()

        # Load vocabulary
        vocab = BanglaVocabulary(config.tokenizer)
        vocab_path = checkpoint_dir / 'vocabulary.json'
        if vocab_path.exists():
            vocab = BanglaVocabulary.load(vocab_path, config.tokenizer)

        # Load model
        model = Wav2VecBertCTCModel(
            config=config,
            vocab_size=len(vocab),
            pretrained_name=str(checkpoint_dir)
        )

        # Create preprocessor
        preprocessor = AudioPreprocessor(config.audio, config.vad)

        # Create decoder
        decoder = CTCDecoder(vocab)

        # CHANGE 3: Construct postprocessor using config fields.
        # Old code hardcoded use_banglabert_detection=False which meant
        # BanglaBERT was never actually used. Now we use the real corrector
        # with model names and tuning knobs pulled from InferenceConfig.
        # CHANGE 4: Field names updated to match new config:
        #   OLD: banglabert_model       → NEW: banglabert_discriminator_model
        #   NEW: banglabert_generator_model (didn't exist before)
        #   NEW: banglabert_discrimination_threshold
        #   NEW: banglabert_max_corrections
        postprocessor = None
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
            model=model,
            vocabulary=vocab,
            preprocessor=preprocessor,
            decoder=decoder,
            postprocessor=postprocessor,
            config=config.inference,
            device=device
        )

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        return_chunks: bool = False,
        use_postprocessing: bool = True
    ) -> Union[str, Dict]:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to audio file
            return_chunks: Whether to return chunk-level transcriptions
            use_postprocessing: Whether to apply BanglaBERT correction

        Returns:
            Transcribed text (or dict with chunks if return_chunks=True)
        """
        # Preprocess audio
        processed = self.preprocessor.process_file(audio_path)

        # Get feature extractor
        feature_extractor = get_feature_extractor(processed.sample_rate)

        all_transcriptions = []
        chunk_results = []

        for chunk_audio, start_time, end_time in processed.chunks:
            features = feature_extractor(
                chunk_audio,
                sampling_rate=processed.sample_rate,
                return_tensors="pt",
                padding=False
            )
            input_features = features.input_features.to(self.device)

            outputs = self.model(input_features=input_features)
            logits = outputs['logits']

            if self.config.decoding_method == 'beam':
                transcription = self.decoder.decode_beam(
                    logits,
                    beam_width=self.config.beam_width
                )[0]
            else:
                transcription = self.decoder.decode_greedy(logits)[0]

            all_transcriptions.append(transcription)

            if return_chunks:
                chunk_results.append({
                    'start': start_time,
                    'end': end_time,
                    'text': transcription
                })

        # Combine all chunks
        full_transcription = ' '.join(all_transcriptions)

        # CHANGE 5: Use .correct() instead of .process().
        # The old postprocessor had a .process() method.
        # The new BanglaBERTPostProcessor from postprocessor.py uses .correct().
        if use_postprocessing and self.postprocessor is not None:
            full_transcription = self.postprocessor.correct(full_transcription)

            if return_chunks:
                for chunk in chunk_results:
                    chunk['text_corrected'] = self.postprocessor.correct(chunk['text'])

        if return_chunks:
            return {
                'text': full_transcription,
                'chunks': chunk_results,
                'duration': processed.original_duration
            }

        return full_transcription

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 8,
        use_postprocessing: bool = True
    ) -> List[str]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for inference
            use_postprocessing: Whether to apply correction

        Returns:
            List of transcriptions
        """
        results = []

        for audio_path in tqdm(audio_paths, desc="Transcribing"):
            try:
                text = self.transcribe(
                    audio_path,
                    use_postprocessing=use_postprocessing
                )
                results.append(text)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append("")

        return results

    def transcribe_dataset(
        self,
        audio_dir: Union[str, Path],
        output_csv: Union[str, Path],
        file_extension: str = '.mp3',
        use_postprocessing: bool = True
    ) -> pd.DataFrame:
        """
        Transcribe all files in a directory.

        Args:
            audio_dir: Directory containing audio files
            output_csv: Path to save results CSV
            file_extension: Audio file extension to look for
            use_postprocessing: Whether to apply correction

        Returns:
            DataFrame with results
        """
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob(f'*{file_extension}'))

        print(f"Found {len(audio_files)} audio files")

        results = []
        for audio_path in tqdm(audio_files, desc="Transcribing"):
            try:
                text = self.transcribe(
                    audio_path,
                    use_postprocessing=use_postprocessing
                )
                results.append({
                    'id': audio_path.stem,
                    'sentence': text
                })
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'id': audio_path.stem,
                    'sentence': ""
                })

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        return df


def generate_submission(
    checkpoint_dir: Union[str, Path],
    test_audio_dir: Union[str, Path],
    sample_submission_path: Union[str, Path],
    output_path: Union[str, Path],
    use_postprocessing: bool = True,
    device: str = "cuda"
):
    """
    Generate competition submission file.

    Args:
        checkpoint_dir: Path to trained model checkpoint
        test_audio_dir: Directory with test audio files
        sample_submission_path: Path to sample submission CSV
        output_path: Path to save submission CSV
        use_postprocessing: Whether to use BanglaBERT correction
        device: Device to use
    """
    inference = ASRInference.from_checkpoint(
        checkpoint_dir,
        use_postprocessing=use_postprocessing,
        device=device
    )

    sample_df = pd.read_csv(sample_submission_path)
    test_ids = sample_df['id'].tolist()

    test_dir = Path(test_audio_dir)

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


def main():
    """Main inference entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Bangla ASR Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, default=None,
                       help='Single audio file to transcribe')
    parser.add_argument('--audio-dir', type=str, default=None,
                       help='Directory of audio files to transcribe')
    parser.add_argument('--output', type=str, default='transcription.csv',
                       help='Output CSV path')
    parser.add_argument('--sample-submission', type=str, default=None,
                       help='Sample submission CSV for competition')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable BanglaBERT post-processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--beam-search', action='store_true',
                       help='Use beam search decoding')
    parser.add_argument('--beam-width', type=int, default=100,
                       help='Beam width for beam search')
    args = parser.parse_args()

    config = get_config()
    if args.beam_search:
        config.inference.decoding_method = 'beam'
        config.inference.beam_width = args.beam_width

    # CHANGE 6: --no-postprocess now correctly disables the new postprocessor.
    # Also set use_banglabert_correction=True in config so from_checkpoint()
    # actually loads the postprocessor when postprocessing is requested.
    use_postprocessing = not args.no_postprocess
    if use_postprocessing:
        config.inference.use_banglabert_correction = True

    if args.sample_submission:
        generate_submission(
            checkpoint_dir=args.checkpoint,
            test_audio_dir=args.audio_dir,
            sample_submission_path=args.sample_submission,
            output_path=args.output,
            use_postprocessing=use_postprocessing,
            device=args.device
        )

    elif args.audio:
        inference = ASRInference.from_checkpoint(
            args.checkpoint,
            use_postprocessing=use_postprocessing,
            device=args.device
        )

        result = inference.transcribe(args.audio, return_chunks=True)

        print(f"\nTranscription: {result['text']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Chunks: {len(result['chunks'])}")

    elif args.audio_dir:
        inference = ASRInference.from_checkpoint(
            args.checkpoint,
            use_postprocessing=use_postprocessing,
            device=args.device
        )

        inference.transcribe_dataset(
            audio_dir=args.audio_dir,
            output_csv=args.output,
            use_postprocessing=use_postprocessing
        )

    else:
        print("Please specify --audio, --audio-dir, or --sample-submission")


if __name__ == "__main__":
    main()
