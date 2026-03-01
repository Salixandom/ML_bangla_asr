"""
Configuration settings for Bangla ASR Pipeline
Model: wav2vec-BERT 2.0 (CTC)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


# =============================================================================
# MODEL PRESETS - Use with --model flag
# =============================================================================
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    'base': {
        'name': 'facebook/w2v-bert-2.0',
        'description': 'Base multilingual wav2vec-BERT 2.0',
        'learning_rate_frozen': 3e-5,
        'learning_rate_unfrozen': 5e-6,
        'warmup_steps': 1000,
    },
    'bangla': {
        'name': 'sazzadul/Shrutimala_Bangla_ASR',
        'description': 'Bangla-finetuned wav2vec-BERT 2.0 (recommended)',
        'learning_rate_frozen': 1e-5,
        'learning_rate_unfrozen': 3e-6,
        'warmup_steps': 500,
    }
}


@dataclass
class AudioConfig:
    """Audio preprocessing configuration"""
    sample_rate: int = 16000
    target_lufs: float = -23.0
    min_segment_duration: float = 0.2
    merge_threshold: float = 0.3
    chunk_min_duration: float = 5.0
    chunk_max_duration: float = 15.0
    chunk_overlap: float = 0.5


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    aggressiveness: int = 2
    frame_duration_ms: int = 30
    padding_duration_ms: int = 300
    min_speech_duration_ms: int = 250


@dataclass
class AugmentationConfig:
    """Training augmentation configuration"""
    speed_perturb_enabled: bool = True
    speed_perturb_prob: float = 0.4
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    bandlimit_enabled: bool = True
    bandlimit_prob: float = 0.3
    bandlimit_target_sr: int = 8000
    volume_perturb_enabled: bool = True
    volume_perturb_prob: float = 0.3
    volume_gain_db_range: tuple = (-6, 6)
    spec_augment_enabled: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "facebook/w2v-bert-2.0"
    ctc_blank_token: str = "<blank>"
    ctc_zero_infinity: bool = True
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    fp16: bool = True
    freeze_feature_encoder: bool = True
    freeze_feature_encoder_steps: int = 0


@dataclass
class DataConfig:
    """Dataset paths and configuration"""
    data_dir: Path = Path("./data")
    train_audio_dir: Path = Path("./data/train")
    test_audio_dir: Path = Path("./data/test")
    examples_dir: Path = Path("./data/examples")
    train_csv: Path = Path("./data/train.csv")
    sample_submission_csv: Path = Path("./data/sample_submission.csv")
    processed_dir: Path = Path("./processed")
    cache_dir: Path = Path("./cache")
    output_dir: Path = Path("./output")
    model_dir: Path = Path("./models")


@dataclass
class TokenizerConfig:
    """Tokenizer configuration for Bangla"""
    vocab_type: str = "char"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    blank_token: str = "<blank>"
    word_delimiter_token: str = "|"
    bpe_vocab_size: int = 1000


@dataclass
class InferenceConfig:
    """Inference configuration"""
    decoding_method: str = "greedy"  # "greedy" or "beam"
    beam_width: int = 100
    lm_weight: float = 0.0

    # ---------------------------------------------------------------------------
    # BanglaBERT post-processing
    # ---------------------------------------------------------------------------
    # CHANGE 1: Default is False — post-processing is for final inference only.
    # Enabling it during training eval would add ~3-5x overhead per eval step.
    # Use --postprocess flag in train.py to enable it for a specific eval run.
    use_banglabert_correction: bool = False

    # CHANGE 2: Both model names needed — discriminator detects bad tokens,
    # generator replaces them. Original config only had the discriminator.
    banglabert_discriminator_model: str = "csebuetnlp/banglabert"
    banglabert_generator_model: str = "csebuetnlp/banglabert_generator"

    # CHANGE 3: Tuning knobs exposed so you can adjust without editing postprocessor.py
    # discrimination_threshold: lower = more aggressive (flag more tokens)
    # max_corrections_per_sentence: safety cap (never correct > X% of words)
    banglabert_discrimination_threshold: float = 0.5
    banglabert_max_corrections: float = 0.3


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all configs"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        """Create directories if they don't exist"""
        for path_field in [self.data.processed_dir, self.data.cache_dir,
                          self.data.output_dir, self.data.model_dir]:
            path_field.mkdir(parents=True, exist_ok=True)


def get_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig()
