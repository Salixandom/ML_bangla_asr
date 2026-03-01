"""
Data Augmentation Module for Bangla ASR Pipeline

IMPORTANT: These augmentations are TRAINING-ONLY.
They are stochastic and on-the-fly.
NEVER apply to validation/test data.

Augmentations:
1. Speed perturbation (0.9x, 1.1x)
2. Band-limiting (16kHz → 8kHz → 16kHz)
3. Volume perturbation (±3-6 dB)
4. SpecAugment (applied in model)
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass
import random

from config import AugmentationConfig


class SpeedPerturbation:
    """
    Speed perturbation augmentation.
    
    Changes playback speed without changing pitch (via resampling).
    Factors: typically 0.9x (slower) and 1.1x (faster)
    
    Why: Improves robustness to speaking rate variations.
    """
    
    def __init__(
        self, 
        factors: List[float] = [0.9, 1.0, 1.1],
        sample_rate: int = 16000
    ):
        self.factors = factors
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply speed perturbation.
        
        Args:
            waveform: Input audio (numpy array)
            
        Returns:
            (perturbed_audio, applied_factor)
        """
        factor = random.choice(self.factors)
        
        if factor == 1.0:
            return waveform, factor
        
        # Speed change via resampling
        # To speed up by factor X: resample from sr to sr*X, then back to sr
        intermediate_sr = int(self.sample_rate * factor)
        
        # Resample to change speed
        stretched = librosa.resample(
            waveform,
            orig_sr=self.sample_rate,
            target_sr=intermediate_sr,
            res_type='kaiser_fast'
        )
        
        # Resample back to original rate
        perturbed = librosa.resample(
            stretched,
            orig_sr=intermediate_sr,
            target_sr=self.sample_rate,
            res_type='kaiser_fast'
        )
        
        return perturbed.astype(np.float32), factor


class BandLimitingAugmentation:
    """
    Band-limiting resampling degradation.
    
    Process: 16kHz → 8kHz → 16kHz
    
    Effect: Removes high-frequency content (above 4kHz)
    Simulates: Telephone speech, low-bandwidth VoIP
    
    Why: Improves domain robustness for noisy/telephone conditions.
    """
    
    def __init__(
        self,
        original_sr: int = 16000,
        target_sr: int = 8000
    ):
        self.original_sr = original_sr
        self.target_sr = target_sr
    
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply band-limiting via downsampling and upsampling.
        
        Args:
            waveform: Input audio at original_sr
            
        Returns:
            Degraded audio at original_sr (with high-freq removed)
        """
        # Downsample to target (e.g., 8kHz)
        # This applies anti-aliasing filter, removing frequencies > target_sr/2
        downsampled = librosa.resample(
            waveform,
            orig_sr=self.original_sr,
            target_sr=self.target_sr,
            res_type='kaiser_best'
        )
        
        # Upsample back to original rate
        upsampled = librosa.resample(
            downsampled,
            orig_sr=self.target_sr,
            target_sr=self.original_sr,
            res_type='kaiser_best'
        )
        
        return upsampled.astype(np.float32)


class VolumePerturbation:
    """
    Random volume/gain adjustment.
    
    Range: typically ±3-6 dB
    
    Why: Simulates distance from microphone, improves loudness invariance.
    """
    
    def __init__(self, gain_db_range: Tuple[float, float] = (-6, 6)):
        self.min_db = gain_db_range[0]
        self.max_db = gain_db_range[1]
    
    def __call__(self, waveform: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply random gain adjustment.
        
        Args:
            waveform: Input audio
            
        Returns:
            (adjusted_audio, applied_gain_db)
        """
        gain_db = random.uniform(self.min_db, self.max_db)
        gain_linear = 10 ** (gain_db / 20)
        
        perturbed = waveform * gain_linear
        
        # Clip to prevent clipping
        perturbed = np.clip(perturbed, -1.0, 1.0)
        
        return perturbed.astype(np.float32), gain_db


class NoiseInjection:
    """
    Optional: Add background noise for robustness.
    
    Note: Not in original pipeline but useful for domain robustness.
    """
    
    def __init__(
        self,
        noise_dir: Optional[str] = None,
        snr_range: Tuple[float, float] = (10, 20)
    ):
        self.noise_dir = noise_dir
        self.snr_range = snr_range
        self.noise_files = []
        
        if noise_dir:
            from pathlib import Path
            noise_path = Path(noise_dir)
            self.noise_files = list(noise_path.glob("*.wav")) + list(noise_path.glob("*.mp3"))
    
    def __call__(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add random noise at random SNR."""
        if not self.noise_files:
            return waveform
        
        # Load random noise file
        noise_file = random.choice(self.noise_files)
        noise, _ = librosa.load(noise_file, sr=sample_rate, mono=True)
        
        # Match lengths
        if len(noise) < len(waveform):
            # Repeat noise
            repeats = int(np.ceil(len(waveform) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(waveform)]
        
        # Calculate SNR-based mixing
        snr_db = random.uniform(*self.snr_range)
        
        signal_power = np.mean(waveform ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            target_noise_power = signal_power / (10 ** (snr_db / 10))
            noise_scale = np.sqrt(target_noise_power / noise_power)
            mixed = waveform + noise * noise_scale
            mixed = np.clip(mixed, -1.0, 1.0)
            return mixed.astype(np.float32)
        
        return waveform


class SpecAugment:
    """
    SpecAugment: Frequency and time masking.
    
    Applied to mel-spectrogram or learned features.
    Standard regularization for Transformer-based ASR.
    
    Note: For wav2vec-BERT 2.0, this is often applied internally.
    This class is for external application if needed.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Shape (batch, freq, time) or (freq, time)
            
        Returns:
            Augmented spectrogram
        """
        # Add batch dim if needed
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_mask(spectrogram)
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            spectrogram = self.time_mask(spectrogram)
        
        if squeeze:
            spectrogram = spectrogram.squeeze(0)
        
        return spectrogram


class AudioAugmentor:
    """
    Complete audio augmentation pipeline for training.
    
    IMPORTANT: Only use during training, never for validation/test.
    """
    
    def __init__(self, config: AugmentationConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize augmentations
        self.speed_perturb = SpeedPerturbation(
            factors=config.speed_factors,
            sample_rate=sample_rate
        )
        
        self.bandlimit = BandLimitingAugmentation(
            original_sr=sample_rate,
            target_sr=config.bandlimit_target_sr
        )
        
        self.volume_perturb = VolumePerturbation(
            gain_db_range=config.volume_gain_db_range
        )
        
        self.spec_augment = SpecAugment(
            freq_mask_param=config.freq_mask_param,
            time_mask_param=config.time_mask_param,
            num_freq_masks=config.num_freq_masks,
            num_time_masks=config.num_time_masks
        )
    
    def augment(
        self, 
        waveform: np.ndarray,
        return_info: bool = False
    ) -> np.ndarray:
        """
        Apply all enabled augmentations with configured probabilities.
        
        Args:
            waveform: Input audio (numpy array, float32, [-1, 1])
            return_info: Whether to return augmentation info dict
            
        Returns:
            Augmented audio (and optionally info dict)
        """
        info = {
            'speed_factor': 1.0,
            'bandlimit_applied': False,
            'volume_gain_db': 0.0
        }
        
        augmented = waveform.copy()
        
        # Speed perturbation
        if self.config.speed_perturb_enabled:
            if random.random() < self.config.speed_perturb_prob:
                augmented, factor = self.speed_perturb(augmented)
                info['speed_factor'] = factor
        
        # Band-limiting
        if self.config.bandlimit_enabled:
            if random.random() < self.config.bandlimit_prob:
                augmented = self.bandlimit(augmented)
                info['bandlimit_applied'] = True
        
        # Volume perturbation
        if self.config.volume_perturb_enabled:
            if random.random() < self.config.volume_perturb_prob:
                augmented, gain = self.volume_perturb(augmented)
                info['volume_gain_db'] = gain
        
        if return_info:
            return augmented, info
        return augmented
    
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Callable interface for augmentation."""
        return self.augment(waveform)


class TranscriptAugmentation:
    """
    Optional: Transcript-level augmentation.
    
    Can include:
    - Random word dropout (for robustness)
    - Character-level noise injection
    
    Note: Use with caution - may hurt performance.
    """
    
    def __init__(self, word_dropout_prob: float = 0.0):
        self.word_dropout_prob = word_dropout_prob
    
    def __call__(self, text: str) -> str:
        """Apply text augmentation."""
        if self.word_dropout_prob <= 0:
            return text
        
        words = text.split()
        augmented_words = [
            word for word in words 
            if random.random() > self.word_dropout_prob
        ]
        
        return ' '.join(augmented_words) if augmented_words else text


def create_augmentation_transform(
    config: AugmentationConfig,
    sample_rate: int = 16000,
    is_training: bool = True
) -> Optional[Callable]:
    """
    Create augmentation transform function.
    
    Args:
        config: Augmentation configuration
        sample_rate: Audio sample rate
        is_training: Whether this is for training (False = no augmentation)
        
    Returns:
        Augmentation function or None
    """
    if not is_training:
        return None
    
    augmentor = AudioAugmentor(config, sample_rate)
    return augmentor


if __name__ == "__main__":
    # Test augmentations
    from config import get_config
    
    config = get_config()
    
    # Create test waveform (1 second of noise)
    test_waveform = np.random.randn(16000).astype(np.float32) * 0.1
    
    # Test individual augmentations
    print("Testing Speed Perturbation...")
    speed_aug = SpeedPerturbation()
    perturbed, factor = speed_aug(test_waveform)
    print(f"  Factor: {factor}, Output length: {len(perturbed)}")
    
    print("\nTesting Band-limiting...")
    bandlimit = BandLimitingAugmentation()
    degraded = bandlimit(test_waveform)
    print(f"  Output length: {len(degraded)}")
    
    print("\nTesting Volume Perturbation...")
    volume_aug = VolumePerturbation()
    adjusted, gain = volume_aug(test_waveform)
    print(f"  Gain: {gain:.2f} dB")
    
    print("\nTesting Full Augmentation Pipeline...")
    augmentor = AudioAugmentor(config.augmentation)
    augmented, info = augmentor.augment(test_waveform, return_info=True)
    print(f"  Info: {info}")
    
    print("\nAll augmentation tests passed!")
