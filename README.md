# Bangla ASR Pipeline

End-to-end Automatic Speech Recognition system for Bangla using **wav2vec-BERT 2.0** with CTC loss.

## Features

- **Model Selection**: Choose between base multilingual or Bangla-pretrained models
- **Flexible Training**: Frozen (fast) or unfrozen (full) encoder training
- **Vocabulary Options**: Custom character vocab or pretrained model vocab
- **wav2vec-BERT 2.0**: State-of-the-art self-supervised speech model
- **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence
- **Robust Preprocessing**: VAD, chunking, loudness normalization with GPU acceleration
- **Multi-threaded Processing**: Parallel audio preprocessing for large datasets
- **Training-only Augmentation**: Speed perturbation, band-limiting, volume perturbation
- **Smart Checkpointing**: Saves best model + latest 3 epoch checkpoints

## Quick Start

```bash
# Step 1: Prepare data
python run.py prepare --input ./data/train.csv --output ./data/train_split.csv

# Step 2: Preprocess audio
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv

# Step 3: Train model (choose one)
python run.py train --model bangla --manifest ./processed/manifest.csv   # Recommended
python run.py train --model base --manifest ./processed/manifest.csv     # Alternative

# Step 4: Generate submission
python run.py submit --checkpoint ./output/best --test-dir ./data/test
```

## Model Selection

### Available Models

| Flag | Model | Description |
|------|-------|-------------|
| `--model base` | `facebook/w2v-bert-2.0` | Base multilingual model (learns Bangla from scratch) |
| `--model bangla` | `sazzadul/Shrutimala_Bangla_ASR` | **Recommended** - Already fine-tuned on Bangla |

### Frozen vs Unfrozen Training

| Mode | Flag | Speed | Use Case |
|------|------|-------|----------|
| **Frozen** (default) | (none) | ~50 min/epoch | Fast training, good results |
| **Unfrozen** | `--unfreeze` | ~24 hrs/epoch | Maximum quality (if you have time) |

**Frozen mode** keeps the first 12 encoder layers fixed, training only:
- Upper 12 encoder layers
- CTC head

**Unfrozen mode** trains all 24 layers (much slower but potentially better).

### Vocabulary Options

| Flag | Description |
|------|-------------|
| (none) | Custom Bangla vocabulary (79 characters) |
| `--use-pretrained-vocab` | Use vocabulary from pretrained model (keeps CTC head) |

## Training Examples

```bash
# Base model, frozen (default)
python run.py train --model base --manifest ./processed/manifest.csv

# Bangla model, frozen (RECOMMENDED)
python run.py train --model bangla --manifest ./processed/manifest.csv

# Bangla model with pretrained vocabulary (fastest convergence)
python run.py train --model bangla --use-pretrained-vocab --manifest ./processed/manifest.csv

# Base model, unfrozen (slow but thorough)
python run.py train --model base --unfreeze --manifest ./processed/manifest.csv

# Bangla model, unfrozen (maximum quality)
python run.py train --model bangla --unfreeze --manifest ./processed/manifest.csv

# Resume from checkpoint
python run.py train --model bangla --manifest ./processed/manifest.csv --resume ./output/epoch_7

# With Weights & Biases logging
python run.py train --model bangla --manifest ./processed/manifest.csv --wandb
```

## Training Configurations

Settings are automatically adjusted based on `--model` and `--unfreeze` flags:

| Config | base | base --unfreeze | bangla | bangla --unfreeze |
|--------|------|-----------------|--------|-------------------|
| Learning Rate | 3e-5 | 5e-6 | 1e-5 | 3e-6 |
| Warmup Steps | 1000 | 1000 | 500 | 500 |
| Frozen Layers | 12 | 0 | 12 | 0 |
| Speed | ~50 min/epoch | ~24 hrs/epoch | ~50 min/epoch | ~24 hrs/epoch |

All model presets are defined in `config.py` under `MODEL_PRESETS`.

## Installation

```bash
# Clone or copy the project
cd bangla_asr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install FFmpeg for faster audio loading
sudo apt install ffmpeg  # Linux/WSL
```

## Project Structure

```
bangla_asr/
├── config.py           # All configurations & MODEL_PRESETS
├── prepare_data.py     # Dataset preparation & splitting
├── preprocessing.py    # Audio preprocessing & VAD
├── augmentation.py     # Training augmentations
├── dataset.py          # Dataset & vocabulary classes
├── train.py            # Training script
├── inference.py        # Inference & submission
├── run.py              # CLI wrapper
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Pipeline Overview

```
Raw MP3/WAV
 → Decode to PCM (16-bit)
 → Resample to 16 kHz
 → Loudness normalization (-23 LUFS)
 → Voice Activity Detection
 → Chunking (5-15 seconds)
 → [Training only] Augmentation
 → Feature extraction (SeamlessM4TFeatureExtractor)
 → wav2vec-BERT 2.0 encoder
 → CTC decoding
 → Text output
```

## Detailed Usage

### 1. Prepare Dataset

```bash
# Full dataset
python run.py prepare --input ./data/train.csv --output ./data/train_split.csv

# Limited dataset (for testing)
python run.py prepare --input ./data/train.csv --output ./data/train_split.csv --limit 10000

# Percentage of data
python run.py prepare --input ./data/train.csv --output ./data/train_split.csv --percentage 5
```

### 2. Preprocess Audio

```bash
# Auto mode (recommended)
python run.py preprocess --audio-dir ./data/train --csv ./data/train_split.csv

# Single-thread with GPU (for smaller datasets)
python run.py preprocess --single-thread

# Custom worker count
python run.py preprocess --workers 4
```

### 3. Train Model

See [Training Examples](#training-examples) above.

**Checkpoints saved:**
```
output/
├── best/           # Best model (lowest WER) - always kept
├── epoch_8/        # Latest 3 epochs kept
├── epoch_9/
└── epoch_10/
```

### 4. Inference

```bash
# Single file
python run.py infer --checkpoint ./output/best --audio test.mp3

# Generate competition submission
python run.py submit --checkpoint ./output/best --test-dir ./data/test \
    --sample-submission ./data/sample_submission.csv --output submission.csv
```

### 5. Python API

```python
from inference import ASRInference

# Load model
asr = ASRInference.from_checkpoint('./output/best', device='cuda')

# Transcribe
text = asr.transcribe('audio.mp3')
print(text)

# With chunk details
result = asr.transcribe('audio.mp3', return_chunks=True)
for chunk in result['chunks']:
    print(f"[{chunk['start']:.2f}s - {chunk['end']:.2f}s]: {chunk['text']}")
```

## Configuration

All settings are in `config.py`. Key sections:

### Model Presets (MODEL_PRESETS)

```python
MODEL_PRESETS = {
    'base': {
        'name': 'facebook/w2v-bert-2.0',
        'learning_rate_frozen': 3e-5,
        'learning_rate_unfrozen': 5e-6,
        'warmup_steps': 1000,
    },
    'bangla': {
        'name': 'sazzadul/Shrutimala_Bangla_ASR',
        'learning_rate_frozen': 1e-5,
        'learning_rate_unfrozen': 3e-6,
        'warmup_steps': 500,
    }
}
```

### Training Settings (ModelConfig)

```python
ModelConfig:
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4   # Effective batch = 32
    num_train_epochs: 30
    fp16: True                       # Mixed precision
    max_grad_norm: 1.0
```

### Audio Processing (AudioConfig)

```python
AudioConfig:
    sample_rate: 16000              # Required for wav2vec-BERT
    chunk_min_duration: 5.0         # Minimum chunk (seconds)
    chunk_max_duration: 15.0        # Maximum chunk (seconds)
```

## Expected Results

| Model | Epochs | WER | CER | Time |
|-------|--------|-----|-----|------|
| base (frozen) | 10 | ~0.40 | ~0.12 | ~8 hrs |
| bangla (frozen) | 10 | ~0.35 | ~0.10 | ~8 hrs |
| bangla + pretrained vocab | 10 | ~0.30 | ~0.08 | ~8 hrs |

## Troubleshooting

### Out of Memory
```python
# In config.py, reduce batch size:
per_device_train_batch_size: 4  # Reduce from 8
gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

### Training Too Slow
- Make sure you're NOT using `--unfreeze` (it's 30x slower)
- Check GPU utilization with `nvidia-smi`
- Use `--model bangla` which has optimized hyperparameters

### WER Not Improving
- Try `--model bangla` instead of `--model base`
- Add `--use-pretrained-vocab` for Bangla model
- Check if data preprocessing completed correctly

### Resume Issues
```bash
# Resume from specific checkpoint
python run.py train --model bangla --manifest ./processed/manifest.csv --resume ./output/epoch_7
```

## Command Reference

| Command | Description |
|---------|-------------|
| `python run.py prepare` | Prepare and split dataset |
| `python run.py preprocess` | Preprocess audio files |
| `python run.py train` | Train the model |
| `python run.py infer` | Run inference on audio |
| `python run.py submit` | Generate submission file |

### Train Command Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model preset (base/bangla) | base |
| `--unfreeze` | Train all encoder layers | False (frozen) |
| `--use-pretrained-vocab` | Use model's vocabulary | False |
| `--manifest` | Path to manifest CSV | Required |
| `--output-dir` | Output directory | ./output |
| `--resume` | Checkpoint to resume from | None |
| `--wandb` | Enable W&B logging | False |

## License

MIT License
