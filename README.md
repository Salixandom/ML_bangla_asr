# Bangla ASR Pipeline

End-to-end Automatic Speech Recognition system for Bangla using **wav2vec-BERT 2.0** with CTC loss.

## Features

- **wav2vec-BERT 2.0**: State-of-the-art self-supervised speech model
- **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence
- **Robust Preprocessing**: VAD, chunking, loudness normalization with GPU acceleration
- **Multi-threaded Processing**: Parallel audio preprocessing for large datasets
- **Training-only Augmentation**: Speed perturbation, band-limiting, volume perturbation
- **Smart Checkpointing**: Saves best model + latest 3 epoch checkpoints
- **Data Limiting**: Train on subsets for quick experimentation

## Pipeline Overview

```
Raw MP3
 → Decode to PCM (librosa for MP3, torchaudio for WAV)
 → Resample to 16 kHz (GPU accelerated)
 → Loudness normalization (-23 LUFS)
 → Amplitude normalization [-1, 1]
 → Voice Activity Detection
 → Speech segmentation
 → Chunking (5-15 seconds)
 → [Training only] Augmentation
 → wav2vec-BERT 2.0 (raw waveform input)
 → CTC decoding
 → Text normalization
```

## Installation

```bash
# Clone or copy the project
cd bangla_asr

# Intall uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# Optional: Install FFmpeg for faster audio loading (Linux/WSL)
sudo apt install ffmpeg

# Optional: Install webrtcvad for better VAD
uv pip install webrtcvad-wheels  # Windows
uv pip install webrtcvad         # Linux/Mac
```

## Project Structure

```
bangla_asr/
├── config.py           # All configuration settings
├── prepare_data.py     # Dataset preparation & splitting
├── preprocessing.py    # Audio preprocessing & VAD (GPU + multi-threaded)
├── augmentation.py     # Training augmentations
├── dataset.py          # Dataset & vocabulary
├── train.py            # Training script with smart checkpointing
├── inference.py        # Inference & submission generation
├── run.py              # CLI wrapper for all commands
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Quick Start

### Using CLI (Recommended)

```bash
# Step 1: Prepare data (with limiting for quick tests)
python run.py prepare --input ./data/train.csv --output ./data/train_split.csv --limit 1000

# Step 2: Preprocess audio
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv

# Step 3: Train model
python run.py train --manifest ./processed/manifest.csv

# Step 4: Generate submission
python run.py submit --checkpoint ./output/best --test-dir ./data/test
```

### Model Run options

```bash
# Base Model (facebook/w2v-bert-2.0)
python run.py train --model base --manifest ./processed/manifest.csv --output-dir ./output

# Bangla Pretrained Model (sazzadul/Shrutimala_Bangla_ASR)
python run.py train --model bangla --manifest ./processed/manifest.csv --output-dir ./output

# Bangla Pretrained Model with Pretrained Vocabulary (sazzadul/Shrutimala_Bangla_ASR)
python run.py train --model bangla --use-pretrained-vocab --manifest ./processed/manifest.csv --output-dir ./output
```

## Detailed Usage

### 1. Prepare Dataset

Organize your data:
```
data/
├── train/              # Training audio (MP3/WAV)
│   ├── audio_001.mp3
│   ├── audio_002.mp3
│   └── ...
├── test/               # Test audio (MP3/WAV)
├── train.csv           # id, sentence columns
└── sample_submission.csv
```

**Split and prepare data:**

```bash
# Use all data
python prepare_data.py --input ./data/train.csv --output ./data/train_split.csv

# Limit to 10,000 samples (for experimentation)
python prepare_data.py --input ./data/train.csv --output ./data/train_split.csv --limit 10000

# Use only 1% of data
python prepare_data.py --input ./data/train.csv --output ./data/train_split.csv --percentage 1

# With audio validation
python prepare_data.py --input ./data/train.csv --output ./data/train_split.csv \
    --limit 5000 --audio-dir ./data/train --validate

# Stratified split by sentence length
python prepare_data.py --input ./data/train.csv --output ./data/train_split.csv --stratify
```

**Data limiting options:**

| Option | Description | Example |
|--------|-------------|---------|
| `--limit N` | Max N samples | `--limit 10000` |
| `--percentage P` | Use P% of data | `--percentage 5` |
| `--valid-ratio R` | Validation split ratio | `--valid-ratio 0.1` |
| `--stratify` | Stratify by length | `--stratify` |

### 2. Preprocess Audio

```bash
# Auto mode (multi-threaded, auto-detect workers)
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv

# Single-thread with GPU (best for smaller datasets)
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv --single-thread

# Custom worker count
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv --workers 4

# Disable GPU
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv --no-gpu
```

**Output:** Creates `processed/manifest.csv` with chunked audio files.

### 3. Train Model

```bash
# Basic training
python run.py train --manifest ./processed/manifest.csv --output-dir ./output

# With Weights & Biases logging
python run.py train --manifest ./processed/manifest.csv --wandb

# Resume from checkpoint
python run.py train --manifest ./processed/manifest.csv --resume ./output/epoch_5
```

**Training features:**
- ✅ Validation after each epoch
- ✅ Saves best model (lowest WER)
- ✅ Keeps only latest 3 epoch checkpoints
- ✅ Mixed precision (FP16) training
- ✅ Gradient accumulation
- ✅ Learning rate warmup

**Checkpoint structure:**
```
output/
├── best/           # Best model (always kept)
├── epoch_28/       # Latest 3 epochs kept
├── epoch_29/
├── epoch_30/
└── step_5000/      # Step checkpoints (if enabled)
```

### 4. Inference

```bash
# Single file
python run.py infer --checkpoint ./output/best --audio test.mp3

# With beam search
python run.py infer --checkpoint ./output/best --audio test.mp3 --beam-width 100

# Generate competition submission
python run.py submit --checkpoint ./output/best --test-dir ./data/test \
    --sample-submission ./data/sample_submission.csv --output submission.csv
```

### 5. Python API

```python
from inference import ASRInference

# Load model
asr = ASRInference.from_checkpoint('./output/best', device='cuda')

# Transcribe single file
text = asr.transcribe('audio.mp3')
print(text)

# Transcribe with chunk details
result = asr.transcribe('audio.mp3', return_chunks=True)
print(f"Full text: {result['text']}")
for chunk in result['chunks']:
    print(f"  [{chunk['start']:.2f}s - {chunk['end']:.2f}s]: {chunk['text']}")

# Batch transcription
texts = asr.transcribe_batch(['file1.mp3', 'file2.mp3'])
```

## Configuration

All settings are in `config.py`. Key parameters:

### Audio Processing
```python
AudioConfig:
    sample_rate: 16000          # wav2vec-BERT requirement
    target_lufs: -23.0          # Loudness target
    chunk_min_duration: 5.0     # Minimum chunk (seconds)
    chunk_max_duration: 15.0    # Maximum chunk (seconds)
```

### Augmentation (Training Only)
```python
AugmentationConfig:
    speed_perturb_prob: 0.4         # 40% chance
    speed_factors: [0.9, 1.0, 1.1]  # Speed variations
    bandlimit_prob: 0.3             # Telephone simulation
    volume_gain_db_range: (-6, 6)   # Volume variation
```

### Model & Training
```python
ModelConfig:
    model_name: "facebook/w2v-bert-2.0"
    learning_rate: 1e-4
    warmup_steps: 500
    num_train_epochs: 30
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4  # Effective batch = 32
    fp16: True                      # Mixed precision
    freeze_feature_encoder: True
    freeze_feature_encoder_steps: 10000
```

## Recommended Training Progression

| Stage | Samples | Command |
|-------|---------|---------|
| Quick test | 1,000 | `--limit 1000` |
| Initial training | 10,000 | `--limit 10000` |
| Validation | 50,000 | `--limit 50000` |
| Serious training | 5% | `--percentage 5` |
| Full training | 100% | (no limit) |

## Important Notes

### wav2vec-BERT 2.0 Input
⚠️ **CRITICAL**: The model expects **RAW WAVEFORM** input, not spectrograms.

```python
# CORRECT: Raw audio
model(input_values=waveform)  # shape: (batch, time_samples)

# WRONG: Don't pass spectrograms
# model(input_values=mel_spectrogram)  # ❌
```

### Augmentation Isolation
⚠️ Augmentations are **ONLY** applied during training, never validation/test.

### Character Vocabulary
- Bangla vowels and consonants (U+0980 to U+09FF)
- Bangla numerals (০-৯)
- Basic punctuation (।,?!.-)
- Special tokens: `<pad>`, `<unk>`, `<blank>`, `|` (word delimiter)

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in config.py
per_device_train_batch_size: 4  # Reduce from 8
gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

### FFmpeg/torchaudio errors on WSL/Linux
```bash
# Install FFmpeg
sudo apt update
sudo apt install ffmpeg

# Or the code will automatically use librosa for MP3 files
```

### webrtcvad Installation Issues
```bash
# Windows (pre-built wheels)
uv pip install webrtcvad-wheels

# Linux
sudo apt-get install python3-dev
uv pip install webrtcvad
```

Falls back to energy-based VAD if webrtcvad unavailable.

### Slow Preprocessing
```bash
# Use more workers
python run.py preprocess --workers 8

# Or use single-thread with GPU for smaller datasets
python run.py preprocess --single-thread
```

## Command Reference

| Command | Description |
|---------|-------------|
| `python run.py prepare` | Prepare and split dataset |
| `python run.py preprocess` | Preprocess audio files |
| `python run.py train` | Train the model |
| `python run.py infer` | Run inference on audio |
| `python run.py submit` | Generate submission file |
| `python run.py demo` | Run pipeline demo |

## License

MIT License