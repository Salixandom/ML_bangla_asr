"""
Bangla ASR Pipeline

End-to-end Automatic Speech Recognition for Bangla using wav2vec-BERT 2.0 with CTC.

Modules:
    - config: Configuration classes
    - preprocessing: Audio preprocessing and VAD
    - augmentation: Training-only data augmentation
    - dataset: PyTorch datasets and vocabulary
    - train: Training loop
    - inference: Inference and post-processing
"""

from .config import (
    PipelineConfig,
    AudioConfig,
    VADConfig,
    AugmentationConfig,
    ModelConfig,
    DataConfig,
    TokenizerConfig,
    InferenceConfig,
    get_config
)

from .preprocessing import (
    AudioPreprocessor,
    TextPreprocessor,
    preprocess_dataset
)

from .augmentation import (
    AudioAugmentor,
    SpeedPerturbation,
    BandLimitingAugmentation,
    VolumePerturbation,
    SpecAugment
)

from .dataset import (
    BanglaVocabulary,
    BanglaASRDataset,
    DataCollatorCTCWithPadding,
    create_dataloaders
)

__version__ = "1.0.0"
__all__ = [
    # Config
    'PipelineConfig', 'AudioConfig', 'VADConfig', 'AugmentationConfig',
    'ModelConfig', 'DataConfig', 'TokenizerConfig', 'InferenceConfig',
    'get_config',
    # Preprocessing
    'AudioPreprocessor', 'TextPreprocessor', 'preprocess_dataset',
    # Augmentation
    'AudioAugmentor', 'SpeedPerturbation', 'BandLimitingAugmentation',
    'VolumePerturbation', 'SpecAugment',
    # Dataset
    'BanglaVocabulary', 'BanglaASRDataset', 'DataCollatorCTCWithPadding',
    'create_dataloaders',
]
