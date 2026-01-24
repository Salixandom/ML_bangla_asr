"""
Configuration settings for Bangla ASR Pipeline
Model: wav2vec-BERT 2.0 (CTC)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio preprocessing configuration"""
    sample_rate: int = 16000  # wav2vec-BERT 2.0 requirement
    target_lufs: float = -23.0  # Loudness normalization target
    min_segment_duration: float = 0.2  # Minimum segment length (seconds)
    merge_threshold: float = 0.3  # Merge segments shorter than this
    chunk_min_duration: float = 5.0  # Minimum chunk length
    chunk_max_duration: float = 15.0  # Maximum chunk length
    chunk_overlap: float = 0.5  # Overlap between chunks (seconds)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    aggressiveness: int = 2  # webrtcvad aggressiveness (0-3)
    frame_duration_ms: int = 30  # Frame size for VAD
    padding_duration_ms: int = 300  # Padding around speech segments
    min_speech_duration_ms: int = 250  # Minimum speech segment


@dataclass
class AugmentationConfig:
    """Training augmentation configuration"""
    # Speed perturbation
    speed_perturb_enabled: bool = True
    speed_perturb_prob: float = 0.4
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    
    # Band-limiting (telephone simulation)
    bandlimit_enabled: bool = True
    bandlimit_prob: float = 0.3
    bandlimit_target_sr: int = 8000
    
    # Volume perturbation
    volume_perturb_enabled: bool = True
    volume_perturb_prob: float = 0.3
    volume_gain_db_range: tuple = (-6, 6)
    
    # SpecAugment (applied in model)
    spec_augment_enabled: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "facebook/w2v-bert-2.0"
    # Alternative: "facebook/wav2vec2-xls-r-300m" for multilingual
    
    # CTC settings
    ctc_blank_token: str = "<blank>"
    ctc_zero_infinity: bool = True
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Batch settings
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Training epochs
    num_train_epochs: int = 5
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Mixed precision
    fp16: bool = True
    
    # Freeze feature encoder for initial training
    freeze_feature_encoder: bool = True
    freeze_feature_encoder_steps: int = 10000


@dataclass
class DataConfig:
    """Dataset paths and configuration"""
    data_dir: Path = Path("./data")
    train_audio_dir: Path = Path("./data/train")
    test_audio_dir: Path = Path("./data/test")
    examples_dir: Path = Path("./data/examples")
    train_csv: Path = Path("./data/train.csv")
    sample_submission_csv: Path = Path("./data/sample_submission.csv")
    
    # Processed data
    processed_dir: Path = Path("./processed")
    cache_dir: Path = Path("./cache")
    
    # Output
    output_dir: Path = Path("./output")
    model_dir: Path = Path("./models")


@dataclass
class TokenizerConfig:
    """Tokenizer configuration for Bangla"""
    # Bangla Unicode range: U+0980 to U+09FF
    vocab_type: str = "char"  # "char" or "bpe"
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    blank_token: str = "<blank>"
    word_delimiter_token: str = "|"
    
    # BPE settings (if using BPE)
    bpe_vocab_size: int = 1000


@dataclass
class InferenceConfig:
    """Inference configuration"""
    decoding_method: str = "greedy"  # "greedy" or "beam"
    beam_width: int = 100
    lm_weight: float = 0.0  # Language model weight (if using)
    
    # BanglaBERT post-processing
    use_banglabert_correction: bool = True
    banglabert_model: str = "csebuetnlp/banglabert"


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
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path_field in [self.data.processed_dir, self.data.cache_dir, 
                          self.data.output_dir, self.data.model_dir]:
            path_field.mkdir(parents=True, exist_ok=True)


# Default configuration instance
def get_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig()
