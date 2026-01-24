"""
Dataset Module for Bangla ASR Pipeline

Provides PyTorch Dataset and DataLoader for wav2vec-BERT 2.0 training.
Handles:
- Loading preprocessed audio
- Feature extraction for wav2vec-BERT 2.0
- On-the-fly augmentation (training only)
- Tokenization
- Batching with dynamic padding

IMPORTANT: wav2vec-BERT 2.0 expects pre-extracted features, NOT raw waveform!
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path
import soundfile as sf
from dataclasses import dataclass

from transformers import SeamlessM4TFeatureExtractor

from config import PipelineConfig, AugmentationConfig, TokenizerConfig
from augmentation import AudioAugmentor, create_augmentation_transform


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


class BanglaVocabulary:
    """
    Character-level vocabulary for Bangla ASR.
    
    Includes:
    - Bangla characters (U+0980 to U+09FF)
    - Bangla juktakkhors (conjunct consonants)
    - Special tokens (pad, unk, blank, word delimiter)
    - Bangla numerals
    - Basic punctuation
    """
    
    # Core Bangla characters (vowels, consonants, modifiers)
    BANGLA_VOWELS = 'অআইঈউঊঋঌএঐওঔ'
    BANGLA_CONSONANTS = 'কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎ'
    BANGLA_VOWEL_SIGNS = 'ািীুূৃৄেৈোৌ'
    BANGLA_MODIFIERS = 'ংঃঁ্ৗ'  # Added ৗ (au length mark)
    BANGLA_NUMERALS = '০১২৩৪৫৬৭৮৯'
    
    # Common Bangla juktakkhors (conjunct consonants / যুক্তাক্ষর)
    # These are frequently used combinations in Bangla text
    BANGLA_JUKTAKKHORS = (
        # ক-based
        'ক্ক ক্ট ক্ত ক্ন ক্ম ক্য ক্র ক্ল ক্ষ ক্স ক্ষ্ম ক্ষ্ণ '
        # খ-based
        'খ্য খ্র '
        # গ-based
        'গ্গ গ্ধ গ্ন গ্ব গ্ম গ্য গ্র গ্ল '
        # ঘ-based
        'ঘ্ন ঘ্য ঘ্র '
        # ঙ-based
        'ঙ্ক ঙ্খ ঙ্গ ঙ্ঘ ঙ্ম '
        # চ-based
        'চ্চ চ্ছ চ্ঞ চ্য '
        # ছ-based
        'ছ্য ছ্র '
        # জ-based
        'জ্জ জ্ঝ জ্ঞ জ্ব জ্য জ্র '
        # ঞ-based
        'ঞ্চ ঞ্ছ ঞ্জ ঞ্ঝ '
        # ট-based
        'ট্ট ট্ব ট্ম ট্য ট্র '
        # ঠ-based
        'ঠ্য ঠ্র '
        # ড-based
        'ড্ড ড্য ড্র '
        # ঢ-based
        'ঢ্য ঢ্র '
        # ণ-based
        'ণ্ট ণ্ঠ ণ্ড ণ্ঢ ণ্ণ ণ্ব ণ্ম ণ্য '
        # ত-based
        'ত্ত ত্থ ত্ন ত্ব ত্ম ত্য ত্র ত্ত্ব '
        # থ-based
        'থ্য থ্র '
        # দ-based
        'দ্দ দ্ধ দ্ব দ্ভ দ্ম দ্য দ্র '
        # ধ-based
        'ধ্ন ধ্ব ধ্ম ধ্য ধ্র '
        # ন-based
        'ন্ট ন্ঠ ন্ড ন্ঢ ন্ত ন্থ ন্দ ন্ধ ন্ন ন্ব ন্ম ন্য ন্র ন্স ন্ত্র ন্ত্য ন্দ্র '
        # প-based
        'প্ট প্ত প্ন প্প প্য প্র প্ল প্স '
        # ফ-based
        'ফ্য ফ্র ফ্ল '
        # ব-based
        'ব্জ ব্দ ব্ধ ব্ব ব্য ব্র ব্ল '
        # ভ-based
        'ভ্য ভ্র '
        # ম-based
        'ম্ন ম্প ম্ফ ম্ব ম্ভ ম্ম ম্য ম্র ম্ল '
        # য-based
        'য্য '
        # র-based (ref/reph combinations)
        'র্ক র্খ র্গ র্ঘ র্চ র্ছ র্জ র্ঝ র্ট র্ঠ র্ড র্ঢ র্ণ '
        'র্ত র্থ র্দ র্ধ র্ন র্প র্ফ র্ব র্ভ র্ম র্য র্ল র্শ র্ষ র্স র্হ '
        # ল-based
        'ল্ক ল্গ ল্ট ল্ড ল্প ল্ফ ল্ব ল্ম ল্য ল্ল '
        # শ-based
        'শ্চ শ্ছ শ্ন শ্ব শ্ম শ্য শ্র শ্ল '
        # ষ-based
        'ষ্ক ষ্ট ষ্ঠ ষ্ণ ষ্প ষ্ফ ষ্ব ষ্ম ষ্য '
        # স-based
        'স্ক স্খ স্ট স্ত স্থ স্ন স্প স্ফ স্ব স্ম স্য স্র স্ল '
        # হ-based
        'হ্ণ হ্ন হ্ব হ্ম হ্য হ্র হ্ল '
        # Special/common combinations
        'ক্ষ্য ঙ্ক্য ঞ্জ্য ত্ত্য দ্ব্য দ্ধ্য ন্ত্য ন্ধ্য স্ত্য স্থ্য '
        'ঞ্চ্য ঞ্জ্ব ন্ত্ব ল্ক্য স্ত্র স্ক্র '
        # More common ones
        'ত্ত্র দ্দ্য দ্ব্র ন্দ্য ন্ত্ব্য স্ত্ব স্ত্ব্য স্থ্য '
    ).split()
    
    # All Bangla characters
    BANGLA_CHARS = (
        BANGLA_VOWELS + 
        BANGLA_CONSONANTS + 
        BANGLA_VOWEL_SIGNS + 
        BANGLA_MODIFIERS + 
        BANGLA_NUMERALS
    )
    
    # Punctuation (minimal)
    PUNCTUATION = '।,?!.-'
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        
        # Special tokens
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token
        self.blank_token = config.blank_token
        self.word_delimiter = config.word_delimiter_token
        
        # Build vocabulary
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character-to-index and index-to-character mappings."""
        # Start with special tokens
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.blank_token: 2,
            self.word_delimiter: 3,  # Space/word boundary
        }
        
        idx = len(self.vocab)
        
        # Add Bangla characters
        for char in self.BANGLA_CHARS:
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1
        
        # Add punctuation
        for char in self.PUNCTUATION:
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1
        
        # Add juktakkhors (conjunct consonants)
        for jukta in self.BANGLA_JUKTAKKHORS:
            if jukta not in self.vocab:
                self.vocab[jukta] = idx
                idx += 1
        
        # Reverse mapping
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        
        # Store special token indices
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.blank_token_id = self.vocab[self.blank_token]
        self.word_delimiter_id = self.vocab[self.word_delimiter]
        
        # Build sorted list of multi-char tokens for longest-match encoding
        self._multi_char_tokens = sorted(
            [t for t in self.vocab.keys() if len(t) > 1 and t not in 
             {self.pad_token, self.unk_token, self.blank_token, self.word_delimiter}],
            key=len, 
            reverse=True  # Longest first for greedy matching
        )
    
    def __len__(self):
        return len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs using longest-match tokenization.
        
        Handles juktakkhors (conjunct consonants) by matching longest tokens first.
        Spaces are converted to word delimiter token.
        Unknown characters are mapped to UNK.
        """
        tokens = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Handle space
            if char == ' ':
                tokens.append(self.word_delimiter_id)
                i += 1
                continue
            
            # Try to match longest multi-char token (juktakkhor) first
            matched = False
            for token in self._multi_char_tokens:
                if text[i:].startswith(token):
                    tokens.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Single character match
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token_id)
            i += 1
        
        return tokens
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special: Skip pad, blank tokens
        """
        chars = []
        special_ids = {self.pad_token_id, self.blank_token_id}
        
        for idx in token_ids:
            if skip_special and idx in special_ids:
                continue
            if idx == self.word_delimiter_id:
                chars.append(' ')
            elif idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        
        return ''.join(chars)
    
    def get_vocab_dict(self) -> Dict[str, int]:
        """Return vocabulary dictionary."""
        return self.vocab.copy()
    
    def save(self, path: Union[str, Path]):
        """Save vocabulary to file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path], config: TokenizerConfig):
        """Load vocabulary from file."""
        import json
        instance = cls(config)
        with open(path, 'r', encoding='utf-8') as f:
            instance.vocab = json.load(f)
        instance.idx_to_char = {v: k for k, v in instance.vocab.items()}
        return instance


class BanglaASRDataset(Dataset):
    """
    PyTorch Dataset for Bangla ASR.
    
    Handles:
    - Loading preprocessed audio chunks
    - On-the-fly augmentation (training only)
    - Tokenization of transcripts
    
    CRITICAL: wav2vec-BERT 2.0 expects RAW WAVEFORM input, not spectrograms.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        vocabulary: BanglaVocabulary,
        split: str = 'train',
        augmentor: Optional[AudioAugmentor] = None,
        max_audio_length: Optional[float] = None,
        max_text_length: Optional[int] = None,
        sample_rate: int = 16000
    ):
        """
        Args:
            manifest_path: Path to CSV manifest with columns:
                          [id, audio_path, sentence, split, duration]
            vocabulary: BanglaVocabulary instance
            split: 'train' or 'valid'
            augmentor: AudioAugmentor for training augmentation
            max_audio_length: Maximum audio duration in seconds
            max_text_length: Maximum transcript length in characters
            sample_rate: Expected sample rate
        """
        self.manifest_path = Path(manifest_path)
        self.vocabulary = vocabulary
        self.split = split
        self.augmentor = augmentor if split == 'train' else None  # Only for training!
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        
        # Load manifest
        self._load_manifest()
    
    def _load_manifest(self):
        """Load and filter manifest."""
        df = pd.read_csv(self.manifest_path)
        
        # Filter by split
        self.data = df[df['split'] == self.split].reset_index(drop=True)
        
        # Filter by duration if specified
        if self.max_audio_length is not None and 'duration' in self.data.columns:
            self.data = self.data[
                self.data['duration'] <= self.max_audio_length
            ].reset_index(drop=True)
        
        # Filter by text length if specified
        if self.max_text_length is not None:
            self.data = self.data[
                self.data['sentence'].str.len() <= self.max_text_length
            ].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} samples for {self.split} split")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - input_features: Extracted features for wav2vec-BERT 2.0
            - labels: Token IDs tensor
            - input_length: Feature sequence length
            - label_length: Label length (tokens)
        """
        row = self.data.iloc[idx]
        
        # Load audio
        audio_path = row['audio_path']
        waveform, sr = sf.read(audio_path, dtype='float32')
        
        # Verify sample rate
        assert sr == self.sample_rate, f"Expected {self.sample_rate}Hz, got {sr}Hz"
        
        # Apply augmentation (training only)
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)
        
        # Extract features for wav2vec-BERT 2.0
        feature_extractor = get_feature_extractor(self.sample_rate)
        features = feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False
        )
        
        # Get input_features: shape (1, seq_len, feature_dim) -> (seq_len, feature_dim)
        input_features = features.input_features.squeeze(0)
        
        # Get transcript and tokenize
        text = row['sentence']
        labels = self.vocabulary.encode(text)
        
        return {
            'input_features': input_features,
            'labels': torch.tensor(labels, dtype=torch.long),
            'input_length': input_features.shape[0],
            'label_length': len(labels),
            'id': row.get('id', str(idx)),
            'text': text
        }


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC training with wav2vec-BERT 2.0.
    
    Handles:
    - Dynamic padding of features to max length in batch
    - Dynamic padding of labels to max length in batch
    - Creating attention masks
    
    NOTE: wav2vec-BERT 2.0 uses input_features (seq_len, feature_dim),
          not input_values (raw waveform).
    """
    
    pad_token_id: int = 0
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            features: List of dicts from dataset __getitem__
            
        Returns:
            Batched tensors with:
            - input_values: Padded features (batch, seq_len, feature_dim)
            - attention_mask: Mask for padded positions
            - labels: Padded labels with -100 for padding
        """
        # Check if we have input_features (wav2vec-BERT 2.0) or input_values (wav2vec2)
        if 'input_features' in features[0]:
            input_key = 'input_features'
            is_2d = True  # Features are (seq_len, feature_dim)
        else:
            input_key = 'input_values'
            is_2d = False  # Raw waveform is (time,)
        
        # Separate inputs and labels
        inputs = [f[input_key] for f in features]
        labels = [f['labels'] for f in features]
        
        if is_2d:
            # Pad 2D features (seq_len, feature_dim)
            max_seq_len = max(x.shape[0] for x in inputs)
            feature_dim = inputs[0].shape[1]
            
            padded_inputs = []
            attention_mask = []
            
            for feat in inputs:
                seq_len = feat.shape[0]
                pad_len = max_seq_len - seq_len
                
                if pad_len > 0:
                    # Pad along sequence dimension
                    padded = torch.nn.functional.pad(feat, (0, 0, 0, pad_len), value=0.0)
                    mask = torch.cat([
                        torch.ones(seq_len),
                        torch.zeros(pad_len)
                    ])
                else:
                    padded = feat
                    mask = torch.ones(seq_len)
                
                padded_inputs.append(padded)
                attention_mask.append(mask)
        else:
            # Pad 1D waveform (time,)
            max_audio_len = max(len(x) for x in inputs)
            padded_inputs = []
            attention_mask = []
            
            for audio in inputs:
                pad_len = max_audio_len - len(audio)
                if pad_len > 0:
                    padded = torch.nn.functional.pad(audio, (0, pad_len), value=0.0)
                    mask = torch.cat([
                        torch.ones(len(audio)),
                        torch.zeros(pad_len)
                    ])
                else:
                    padded = audio
                    mask = torch.ones(len(audio))
                padded_inputs.append(padded)
                attention_mask.append(mask)
        
        # Pad labels (use -100 for CTC ignore index)
        max_label_len = max(len(x) for x in labels)
        padded_labels = []
        
        for label in labels:
            pad_len = max_label_len - len(label)
            if pad_len > 0:
                padded = torch.nn.functional.pad(label, (0, pad_len), value=-100)
            else:
                padded = label
            padded_labels.append(padded)
        
        # Stack into batch
        # NOTE: We use 'input_values' key for compatibility with transformers
        batch = {
            'input_values': torch.stack(padded_inputs),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(padded_labels),
            'input_lengths': torch.tensor([f['input_length'] for f in features]),
            'label_lengths': torch.tensor([f['label_length'] for f in features])
        }
        
        return batch


def create_dataloaders(
    config: PipelineConfig,
    vocabulary: BanglaVocabulary,
    manifest_path: Union[str, Path],
    batch_size: Optional[int] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Pipeline configuration
        vocabulary: BanglaVocabulary instance
        manifest_path: Path to processed manifest CSV
        batch_size: Override batch size from config
        num_workers: Number of dataloader workers
        
    Returns:
        Dictionary with 'train' and 'valid' DataLoaders
    """
    batch_size = batch_size or config.model.per_device_train_batch_size
    
    # Create augmentor for training
    train_augmentor = AudioAugmentor(
        config.augmentation,
        sample_rate=config.audio.sample_rate
    )
    
    # Create datasets
    train_dataset = BanglaASRDataset(
        manifest_path=manifest_path,
        vocabulary=vocabulary,
        split='train',
        augmentor=train_augmentor,
        max_audio_length=config.audio.chunk_max_duration,
        sample_rate=config.audio.sample_rate
    )
    
    valid_dataset = BanglaASRDataset(
        manifest_path=manifest_path,
        vocabulary=vocabulary,
        split='valid',
        augmentor=None,  # No augmentation for validation!
        max_audio_length=config.audio.chunk_max_duration,
        sample_rate=config.audio.sample_rate
    )
    
    # Create data collator
    collator = DataCollatorCTCWithPadding(
        pad_token_id=vocabulary.pad_token_id
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'valid': valid_loader
    }


class HuggingFaceDatasetWrapper:
    """
    Wrapper to create HuggingFace-compatible dataset.
    
    Useful for using with HuggingFace Trainer.
    """
    
    @staticmethod
    def from_manifest(
        manifest_path: Union[str, Path],
        vocabulary: BanglaVocabulary,
        split: str = 'train',
        augmentation_config: Optional[AugmentationConfig] = None,
        sample_rate: int = 16000
    ):
        """
        Create HuggingFace Dataset from manifest.
        """
        from datasets import Dataset as HFDataset, Audio
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        df = df[df['split'] == split].reset_index(drop=True)
        
        # Create HF dataset
        hf_dataset = HFDataset.from_pandas(df)
        
        # Add audio column
        hf_dataset = hf_dataset.cast_column(
            'audio_path',
            Audio(sampling_rate=sample_rate)
        )
        
        return hf_dataset
    
    @staticmethod
    def prepare_dataset(
        batch,
        vocabulary: BanglaVocabulary,
        processor=None
    ):
        """
        Prepare batch for training (used with dataset.map()).
        """
        # Get audio
        audio = batch['audio_path']
        
        # Process with wav2vec processor if available
        if processor is not None:
            inputs = processor(
                audio['array'],
                sampling_rate=audio['sampling_rate'],
                return_tensors='pt'
            )
            batch['input_values'] = inputs.input_values[0]
        else:
            batch['input_values'] = audio['array']
        
        # Tokenize text
        batch['labels'] = vocabulary.encode(batch['sentence'])
        
        return batch


if __name__ == "__main__":
    # Test dataset components
    from config import get_config
    
    config = get_config()
    
    # Create vocabulary
    vocab = BanglaVocabulary(config.tokenizer)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test encoding/decoding
    test_text = "আমি বাংলায় কথা বলি"
    encoded = vocab.encode(test_text)
    decoded = vocab.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Save vocabulary
    vocab.save(config.data.processed_dir / 'vocabulary.json')
    print(f"Vocabulary saved to {config.data.processed_dir / 'vocabulary.json'}")
