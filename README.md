# Bangla ASR Pipeline

End-to-end Automatic Speech Recognition system for Bangla using **wav2vec-BERT 2.0** with CTC loss.

## Features

- **wav2vec-BERT 2.0**: State-of-the-art self-supervised speech model
- **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence
- **Robust Preprocessing**: VAD, chunking, loudness normalization
- **Training-only Augmentation**: Speed perturbation, band-limiting, volume perturbation
- **BanglaBERT Post-processing**: Spelling correction and text normalization

## Pipeline Overview

```
Raw MP3
 → Decode to PCM
 → Resample to 16 kHz
 → Loudness normalization (-23 LUFS)
 → Amplitude normalization [-1, 1]
 → Voice Activity Detection
 → Speech segmentation
 → Chunking (5-15 seconds)
 → [Training only] Augmentation
 → wav2vec-BERT 2.0 (raw waveform input)
 → CTC decoding
 → BanglaBERT text normalization
```

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

# Optional: Install webrtcvad (may need system dependencies)
pip install webrtcvad
```

## Project Structure

```
bangla_asr/
├── config.py           # All configuration settings
├── preprocessing.py    # Audio preprocessing & VAD
├── augmentation.py     # Training augmentations
├── dataset.py          # Dataset & vocabulary
├── train.py            # Training script
├── inference.py        # Inference & submission
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Usage

### 1. Prepare Dataset

Organize your data:
```
data/
├── train/              # Training audio (MP3)
├── test/               # Test audio (MP3)
├── train.csv           # id, sentence, split columns
└── sample_submission.csv
```

### 2. Preprocess Data

```python
from preprocessing import preprocess_dataset
from config import get_config

config = get_config()

# Preprocess all audio files
preprocess_dataset(
    audio_dir=config.data.train_audio_dir,
    csv_path=config.data.train_csv,
    output_dir=config.data.processed_dir,
    audio_config=config.audio,
    vad_config=config.vad
)
```

Or run as script:
```bash
python preprocessing.py
```

### 3. Train Model

```bash
# Basic training
python train.py \
    --manifest ./processed/manifest.csv \
    --output-dir ./output

# With Weights & Biases logging
python train.py \
    --manifest ./processed/manifest.csv \
    --output-dir ./output \
    --wandb \
    --wandb-project bangla-asr

# Resume from checkpoint
python train.py \
    --manifest ./processed/manifest.csv \
    --output-dir ./output \
    --resume ./output/epoch_5
```

### 4. Inference

```bash
# Single file
python inference.py \
    --checkpoint ./output/best \
    --audio test_audio.mp3

# Directory of files
python inference.py \
    --checkpoint ./output/best \
    --audio-dir ./data/test \
    --output transcriptions.csv

# Generate competition submission
python inference.py \
    --checkpoint ./output/best \
    --audio-dir ./data/test \
    --sample-submission ./data/sample_submission.csv \
    --output submission.csv

# Use beam search decoding
python inference.py \
    --checkpoint ./output/best \
    --audio test.mp3 \
    --beam-search \
    --beam-width 100
```

### 5. Python API

```python
from inference import ASRInference

# Load model
inference = ASRInference.from_checkpoint(
    checkpoint_dir='./output/best',
    use_postprocessing=True,
    device='cuda'
)

# Transcribe single file
text = inference.transcribe('audio.mp3')
print(text)

# Transcribe with chunk details
result = inference.transcribe('audio.mp3', return_chunks=True)
print(f"Full text: {result['text']}")
print(f"Duration: {result['duration']:.2f}s")
for chunk in result['chunks']:
    print(f"  [{chunk['start']:.2f}-{chunk['end']:.2f}]: {chunk['text']}")

# Batch transcription
texts = inference.transcribe_batch(['file1.mp3', 'file2.mp3', 'file3.mp3'])
```

## Configuration

All settings are in `config.py`. Key parameters:

### Audio Processing
```python
AudioConfig:
    sample_rate: 16000      # wav2vec-BERT requirement
    target_lufs: -23.0      # Loudness target
    chunk_min_duration: 5.0  # Seconds
    chunk_max_duration: 15.0
```

### Augmentation (Training Only)
```python
AugmentationConfig:
    speed_perturb_prob: 0.4     # 40% chance
    speed_factors: [0.9, 1.0, 1.1]
    bandlimit_prob: 0.3         # Telephone simulation
    volume_gain_db_range: (-6, 6)
```

### Model
```python
ModelConfig:
    model_name: "facebook/w2v-bert-2.0"
    learning_rate: 1e-4
    warmup_steps: 500
    num_train_epochs: 30
    freeze_feature_encoder: True
    freeze_feature_encoder_steps: 10000
```

## Important Notes

### wav2vec-BERT 2.0 Input
⚠️ **CRITICAL**: The model expects **RAW WAVEFORM** input, not spectrograms or MFCCs.

```python
# CORRECT: Raw audio
model(input_values=waveform)  # shape: (batch, time)

# WRONG: Don't pass spectrograms
# model(input_values=mel_spectrogram)  # ❌
```

### Augmentation Isolation
⚠️ Augmentations are **ONLY** applied during training, never validation/test.

```python
# Training: augmentation enabled
train_dataset = BanglaASRDataset(split='train', augmentor=augmentor)

# Validation: NO augmentation
valid_dataset = BanglaASRDataset(split='valid', augmentor=None)
```

### Character Vocabulary
The default vocabulary includes:
- Bangla vowels and consonants (U+0980 to U+09FF)
- Bangla numerals (০-৯)
- Basic punctuation (।,?!.-)
- Special tokens: `<pad>`, `<unk>`, `<blank>`, `|` (word delimiter)

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable `fp16` training

### webrtcvad Installation Issues
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Then install
pip install webrtcvad
```

Falls back to energy-based VAD if webrtcvad unavailable.

### No GPU Available
```python
# In config.py or command line
device: str = "cpu"
```

## Citation

If you use this code, please cite:

```bibtex
@misc{bangla_asr_pipeline,
  title={Bangla ASR Pipeline with wav2vec-BERT 2.0},
  year={2024}
}
```

## License

MIT License
