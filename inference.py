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

from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
)

from config import PipelineConfig, InferenceConfig, get_config
from preprocessing import AudioPreprocessor
from dataset import BanglaVocabulary
from train import Wav2VecBertCTCModel, CTCDecoder


class BanglaBERTPostProcessor:
    """
    Post-processing for Bangla ASR output.
    
    Uses:
    1. csebuetnlp/normalizer for text normalization
    2. BanglaBERT (ELECTRA discriminator) for detecting fake/incorrect tokens
    3. BanglaT5 for actual correction (optional)
    
    Note: BanglaBERT is an ELECTRA-style model (discriminator), 
    NOT a masked language model. It detects replaced tokens, not fills masks.
    """
    
    def __init__(
        self,
        use_normalizer: bool = True,
        use_banglabert_detection: bool = False,  # For detecting errors
        use_banglat5_correction: bool = False,   # For actual correction
        device: str = "cuda"
    ):
        self.device = device
        self.use_normalizer = use_normalizer
        self.use_banglabert_detection = use_banglabert_detection
        self.use_banglat5_correction = use_banglat5_correction
        
        # Load normalizer (required for BanglaBERT)
        self.normalizer = None
        if use_normalizer:
            try:
                from normalizer import normalize
                self.normalizer = normalize
                print("Loaded csebuetnlp normalizer")
            except ImportError:
                print("WARNING: normalizer not installed!")
                print("Install with: pip install git+https://github.com/csebuetnlp/normalizer")
                self.normalizer = None
        
        # Load BanglaBERT for error detection (ELECTRA discriminator)
        self.banglabert_model = None
        self.banglabert_tokenizer = None
        if use_banglabert_detection:
            try:
                from transformers import AutoModelForPreTraining, AutoTokenizer
                self.banglabert_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
                self.banglabert_model = AutoModelForPreTraining.from_pretrained("csebuetnlp/banglabert")
                self.banglabert_model.to(device)
                self.banglabert_model.eval()
                print("Loaded BanglaBERT discriminator for error detection")
            except Exception as e:
                print(f"WARNING: Could not load BanglaBERT: {e}")
        
        # Load BanglaT5 for seq2seq correction (optional)
        self.banglat5_model = None
        self.banglat5_tokenizer = None
        if use_banglat5_correction:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self.banglat5_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5")
                self.banglat5_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5")
                self.banglat5_model.to(device)
                self.banglat5_model.eval()
                print("Loaded BanglaT5 for text correction")
            except Exception as e:
                print(f"WARNING: Could not load BanglaT5: {e}")
    
    def normalize_text(self, text: str) -> str:
        """
        Apply csebuetnlp normalizer.
        This is REQUIRED before using any csebuetnlp model.
        """
        if self.normalizer is None:
            return text
        
        try:
            return self.normalizer(text)
        except Exception as e:
            print(f"Normalization error: {e}")
            return text
    
    def detect_errors(self, text: str) -> list:
        """
        Use BanglaBERT (ELECTRA) to detect potentially incorrect tokens.
        
        Returns list of (token, is_suspicious) tuples.
        BanglaBERT outputs 1 for "replaced/fake" tokens, 0 for "original" tokens.
        """
        if self.banglabert_model is None:
            return []
        
        try:
            import torch
            
            # Normalize first (required!)
            normalized_text = self.normalize_text(text)
            
            # Tokenize
            tokens = self.banglabert_tokenizer.tokenize(normalized_text)
            inputs = self.banglabert_tokenizer.encode(normalized_text, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Get discriminator predictions
            with torch.no_grad():
                outputs = self.banglabert_model(inputs).logits
                predictions = torch.round((torch.sign(outputs) + 1) / 2)
            
            # Map predictions to tokens (exclude [CLS] and [SEP])
            pred_list = predictions.squeeze().tolist()[1:-1]
            
            results = []
            for token, pred in zip(tokens, pred_list):
                is_suspicious = int(pred) == 1  # 1 = likely replaced/incorrect
                results.append((token, is_suspicious))
            
            return results
            
        except Exception as e:
            print(f"Error detection failed: {e}")
            return []
    
    def correct_with_t5(self, text: str) -> str:
        """
        Use BanglaT5 for seq2seq text correction.
        """
        if self.banglat5_model is None:
            return text
        
        try:
            import torch
            
            # Normalize first
            normalized_text = self.normalize_text(text)
            
            # Encode
            inputs = self.banglat5_tokenizer.encode(
                normalized_text, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.banglat5_model.generate(
                    inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            corrected = self.banglat5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected
            
        except Exception as e:
            print(f"T5 correction failed: {e}")
            return text
    
    def clean_asr_output(self, text: str) -> str:
        """
        Basic cleaning for ASR output:
        - Remove repeated characters/words
        - Fix spacing around punctuation
        - Remove artifacts
        """
        import re
        
        # Remove excessive character repetition (e.g., "আআআমি" -> "আমি")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove excessive word repetition
        words = text.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
            prev_word = word
        text = ' '.join(cleaned_words)
        
        # Fix punctuation spacing
        for punct in '।,?!.':
            text = text.replace(f' {punct}', punct)
            text = text.replace(f'{punct}', f'{punct} ')
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def process(self, text: str, correct_spelling: bool = True) -> str:
        """
        Full post-processing pipeline.
        
        Args:
            text: Raw ASR output
            correct_spelling: Whether to apply correction
            
        Returns:
            Cleaned and normalized text
        """
        # Step 1: Basic ASR output cleaning
        text = self.clean_asr_output(text)
        
        # Step 2: Apply normalizer (handles Unicode normalization, etc.)
        text = self.normalize_text(text)
        
        # Step 3: Optional T5-based correction
        if correct_spelling and self.banglat5_model is not None:
            text = self.correct_with_t5(text)
        
        # Final cleanup
        text = ' '.join(text.split())
        
        return text.strip()


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
        
        # Move model to device
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
        
        # Create postprocessor
        postprocessor = None
        if use_postprocessing:
            try:
                postprocessor = BanglaBERTPostProcessor(
                    use_normalizer=True,
                    use_banglabert_detection=False,  # Detection only, not correction
                    use_banglat5_correction=False,   # Enable if you want T5 correction
                )
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
        
        all_transcriptions = []
        chunk_results = []
        
        for chunk_audio, start_time, end_time in processed.chunks:
            # Prepare input
            input_values = torch.from_numpy(chunk_audio).float().unsqueeze(0)
            input_values = input_values.to(self.device)
            
            # Forward pass
            outputs = self.model(input_values=input_values)
            logits = outputs['logits']
            
            # Decode
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
        
        # Apply post-processing
        if use_postprocessing and self.postprocessor is not None:
            full_transcription = self.postprocessor.process(full_transcription)
            
            if return_chunks:
                for chunk in chunk_results:
                    chunk['text_corrected'] = self.postprocessor.process(chunk['text'])
        
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
        
        # Save results
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
    # Load inference pipeline
    inference = ASRInference.from_checkpoint(
        checkpoint_dir,
        use_postprocessing=use_postprocessing,
        device=device
    )
    
    # Load sample submission to get IDs
    sample_df = pd.read_csv(sample_submission_path)
    test_ids = sample_df['id'].tolist()
    
    # Find audio files
    test_dir = Path(test_audio_dir)
    
    results = []
    for test_id in tqdm(test_ids, desc="Generating submission"):
        # Find audio file
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
    
    # Save submission
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
    
    # Update config if using beam search
    config = get_config()
    if args.beam_search:
        config.inference.decoding_method = 'beam'
        config.inference.beam_width = args.beam_width
    
    use_postprocessing = not args.no_postprocess
    
    if args.sample_submission:
        # Competition submission mode
        generate_submission(
            checkpoint_dir=args.checkpoint,
            test_audio_dir=args.audio_dir,
            sample_submission_path=args.sample_submission,
            output_path=args.output,
            use_postprocessing=use_postprocessing,
            device=args.device
        )
    
    elif args.audio:
        # Single file mode
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
        # Directory mode
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