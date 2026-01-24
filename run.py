#!/usr/bin/env python
"""
Bangla ASR Pipeline - Main Entry Point

Provides convenient commands for the complete pipeline:
- preprocess: Preprocess audio dataset
- train: Train the model
- infer: Run inference
- submit: Generate competition submission

Usage:
    python run.py preprocess --data-dir ./data
    python run.py train --manifest ./processed/manifest.csv
    python run.py infer --checkpoint ./output/best --audio test.mp3
    python run.py submit --checkpoint ./output/best --test-dir ./data/test
"""

import argparse
import sys
from pathlib import Path


def preprocess_command(args):
    """Run preprocessing pipeline."""
    from preprocessing import preprocess_dataset
    from config import get_config
    
    config = get_config()
    
    # Override paths if provided
    audio_dir = Path(args.audio_dir) if args.audio_dir else config.data.train_audio_dir
    csv_path = Path(args.csv) if args.csv else config.data.train_csv
    output_dir = Path(args.output) if args.output else config.data.processed_dir
    
    # Determine workers
    if args.single_thread:
        num_workers = 1
    else:
        num_workers = args.workers  # None = auto-detect
    
    use_gpu = not args.no_gpu
    
    print(f"{'='*60}")
    print(f"PREPROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Audio directory: {audio_dir}")
    print(f"CSV manifest: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {num_workers if num_workers else 'auto'}")
    print(f"GPU acceleration: {use_gpu}")
    print(f"{'='*60}\n")
    
    preprocess_dataset(
        audio_dir=audio_dir,
        csv_path=csv_path,
        output_dir=output_dir,
        audio_config=config.audio,
        vad_config=config.vad,
        num_workers=num_workers,
        use_gpu=use_gpu
    )
    
    print("\n✅ Preprocessing complete!")


def prepare_command(args):
    """Run data preparation and splitting."""
    import subprocess
    
    cmd = [
        sys.executable, 'prepare_data.py',
        '--input', args.input,
        '--output', args.output,
        '--valid-ratio', str(args.valid_ratio),
        '--seed', str(args.seed),
    ]
    
    if args.audio_dir:
        cmd.extend(['--audio-dir', args.audio_dir])
        cmd.append('--validate')
    
    if args.stratify:
        cmd.append('--stratify')
    
    # Data limiting options
    if args.limit:
        cmd.extend(['--limit', str(args.limit)])
    
    if args.percentage:
        cmd.extend(['--percentage', str(args.percentage)])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def train_command(args):
    """Run training."""
    import subprocess
    
    cmd = [
        sys.executable, 'train.py',
        '--manifest', args.manifest,
        '--output-dir', args.output_dir
    ]
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
    
    if args.wandb:
        cmd.append('--wandb')
        if args.wandb_project:
            cmd.extend(['--wandb-project', args.wandb_project])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def infer_command(args):
    """Run inference."""
    from inference import ASRInference
    
    inference = ASRInference.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        use_postprocessing=not args.no_postprocess,
        device=args.device
    )
    
    if args.audio:
        result = inference.transcribe(args.audio, return_chunks=True)
        print(f"\n{'='*60}")
        print(f"Transcription: {result['text']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Chunks: {len(result['chunks'])}")
        print(f"{'='*60}")
    
    elif args.audio_dir:
        inference.transcribe_dataset(
            audio_dir=args.audio_dir,
            output_csv=args.output,
            use_postprocessing=not args.no_postprocess
        )


def submit_command(args):
    """Generate competition submission."""
    from inference import generate_submission
    
    generate_submission(
        checkpoint_dir=args.checkpoint,
        test_audio_dir=args.test_dir,
        sample_submission_path=args.sample_submission,
        output_path=args.output,
        use_postprocessing=not args.no_postprocess,
        device=args.device
    )


def demo_command(args):
    """Run a quick demo with sample data."""
    import numpy as np
    import soundfile as sf
    from config import get_config
    from preprocessing import AudioPreprocessor
    from augmentation import AudioAugmentor
    from dataset import BanglaVocabulary
    
    config = get_config()
    
    print("="*60)
    print("Bangla ASR Pipeline Demo")
    print("="*60)
    
    # Test vocabulary
    print("\n1. Testing Vocabulary...")
    vocab = BanglaVocabulary(config.tokenizer)
    test_text = "আমি বাংলায় কথা বলি"
    encoded = vocab.encode(test_text)
    decoded = vocab.decode(encoded)
    print(f"   Original: {test_text}")
    print(f"   Encoded:  {encoded[:10]}...")
    print(f"   Decoded:  {decoded}")
    print(f"   Vocab size: {len(vocab)}")
    
    # Test preprocessing
    print("\n2. Testing Audio Preprocessing...")
    preprocessor = AudioPreprocessor(config.audio, config.vad)
    print(f"   Sample rate: {config.audio.sample_rate}")
    print(f"   Target LUFS: {config.audio.target_lufs}")
    print(f"   Chunk duration: {config.audio.chunk_min_duration}-{config.audio.chunk_max_duration}s")
    
    # Test augmentation
    print("\n3. Testing Augmentation...")
    augmentor = AudioAugmentor(config.augmentation, sample_rate=config.audio.sample_rate)
    test_waveform = np.random.randn(config.audio.sample_rate * 5).astype(np.float32) * 0.1
    augmented, info = augmentor.augment(test_waveform, return_info=True)
    print(f"   Speed factor: {info['speed_factor']}")
    print(f"   Band-limit applied: {info['bandlimit_applied']}")
    print(f"   Volume gain: {info['volume_gain_db']:.2f} dB")
    
    print("\n" + "="*60)
    print("Demo complete! All components working correctly.")
    print("="*60)
    
    # Save vocabulary
    vocab_path = config.data.processed_dir / 'vocabulary.json'
    vocab.save(vocab_path)
    print(f"\nVocabulary saved to: {vocab_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Bangla ASR Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Step 1: Prepare and split your raw data
    python run.py prepare --input raw_transcripts.csv --output data/train.csv --audio-dir data/train

    # Step 2: Preprocess audio (VAD, chunking, normalization)
    python run.py preprocess --audio-dir ./data/train --csv ./data/train.csv

    # Step 3: Train model
    python run.py train --manifest ./processed/manifest.csv --wandb

    # Step 4: Run inference
    python run.py infer --checkpoint ./output/best --audio test.mp3

    # Step 5: Generate submission
    python run.py submit --checkpoint ./output/best --test-dir ./data/test

    # Run demo
    python run.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prepare command (NEW)
    prepare_parser = subparsers.add_parser('prepare', help='Prepare and split dataset')
    prepare_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input CSV/JSON with transcriptions')
    prepare_parser.add_argument('--output', '-o', type=str, default='./data/train.csv',
                               help='Output CSV path')
    prepare_parser.add_argument('--audio-dir', '-a', type=str, default=None,
                               help='Audio directory (validates files exist)')
    prepare_parser.add_argument('--valid-ratio', '-v', type=float, default=0.1,
                               help='Validation ratio (default: 0.1)')
    prepare_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed')
    prepare_parser.add_argument('--stratify', action='store_true',
                               help='Stratify by sentence length')
    # Data limiting options
    prepare_parser.add_argument('--limit', '-l', type=int, default=None,
                               help='Max samples (e.g., 1000, 10000)')
    prepare_parser.add_argument('--percentage', '-p', type=float, default=None,
                               help='Percentage of data (e.g., 1, 5, 10)')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess audio dataset')
    preprocess_parser.add_argument('--audio-dir', type=str, help='Audio directory')
    preprocess_parser.add_argument('--csv', type=str, help='CSV manifest path')
    preprocess_parser.add_argument('--output', type=str, help='Output directory')
    preprocess_parser.add_argument('--workers', type=int, default=None, 
                                   help='Number of parallel workers (default: auto)')
    preprocess_parser.add_argument('--no-gpu', action='store_true',
                                   help='Disable GPU acceleration')
    preprocess_parser.add_argument('--single-thread', action='store_true',
                                   help='Use single thread with GPU (best for small datasets)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--manifest', type=str, required=True, help='Processed manifest CSV')
    train_parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    train_parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    train_parser.add_argument('--wandb', action='store_true', help='Use W&B logging')
    train_parser.add_argument('--wandb-project', type=str, default='bangla-asr', help='W&B project')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    infer_parser.add_argument('--audio', type=str, help='Single audio file')
    infer_parser.add_argument('--audio-dir', type=str, help='Directory of audio files')
    infer_parser.add_argument('--output', type=str, default='transcriptions.csv', help='Output CSV')
    infer_parser.add_argument('--no-postprocess', action='store_true', help='Disable BanglaBERT')
    infer_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Generate submission')
    submit_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    submit_parser.add_argument('--test-dir', type=str, required=True, help='Test audio directory')
    submit_parser.add_argument('--sample-submission', type=str, 
                              default='./data/sample_submission.csv', help='Sample submission')
    submit_parser.add_argument('--output', type=str, default='submission.csv', help='Output path')
    submit_parser.add_argument('--no-postprocess', action='store_true', help='Disable BanglaBERT')
    submit_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run pipeline demo')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_command(args)
    elif args.command == 'preprocess':
        preprocess_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'infer':
        infer_command(args)
    elif args.command == 'submit':
        submit_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
