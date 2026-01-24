"""
Data Preparation Script for Bangla ASR Pipeline

Features:
1. Split dataset into train/valid
2. Validate audio files exist
3. Basic statistics
4. Handle different input formats
5. Stratified splitting option

Usage:
    python prepare_data.py --input raw_data.csv --output data/train.csv --valid-ratio 0.1
    python prepare_data.py --audio-dir data/train --input transcripts.csv --output data/train.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from collections import Counter
import random
import json


def load_input_data(input_path: Path, audio_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load input data from various formats.
    
    Supports:
    - CSV with 'id' and 'sentence' columns
    - CSV with 'audio_path' and 'transcription' columns
    - JSON/JSONL format
    
    Returns DataFrame with standardized columns: id, sentence
    """
    suffix = input_path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(input_path)
    elif suffix == '.json':
        df = pd.read_json(input_path)
    elif suffix == '.jsonl':
        df = pd.read_json(input_path, lines=True)
    elif suffix == '.tsv':
        df = pd.read_csv(input_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Standardize column names
    column_mapping = {
        # Common variations for ID
        'audio_id': 'id',
        'file_id': 'id',
        'filename': 'id',
        'file_name': 'id',
        'audio_path': 'id',
        'path': 'id',
        'audio': 'id',
        
        # Common variations for transcription
        'transcription': 'sentence',
        'transcript': 'sentence',
        'text': 'sentence',
        'label': 'sentence',
        'target': 'sentence',
        'ground_truth': 'sentence',
        'bangla_text': 'sentence',
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure required columns exist
    if 'id' not in df.columns or 'sentence' not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find 'id' and 'sentence' columns. Please check column names.")
    
    # Clean up IDs (remove extension if present)
    df['id'] = df['id'].apply(lambda x: Path(str(x)).stem)
    
    # Remove any existing split column (we'll create our own)
    if 'split' in df.columns:
        print("Warning: Existing 'split' column will be overwritten")
        df = df.drop(columns=['split'])
    
    return df[['id', 'sentence']]


def validate_audio_files(
    df: pd.DataFrame, 
    audio_dir: Path,
    extensions: List[str] = ['.mp3', '.wav', '.flac', '.ogg']
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate that audio files exist for each ID.
    
    Returns:
        - DataFrame with only valid entries
        - List of missing file IDs
    """
    valid_ids = []
    missing_ids = []
    
    for idx, row in df.iterrows():
        audio_id = row['id']
        found = False
        
        for ext in extensions:
            audio_path = audio_dir / f"{audio_id}{ext}"
            if audio_path.exists():
                found = True
                break
        
        if found:
            valid_ids.append(idx)
        else:
            missing_ids.append(audio_id)
    
    valid_df = df.loc[valid_ids].reset_index(drop=True)
    
    return valid_df, missing_ids


def compute_text_stats(df: pd.DataFrame) -> dict:
    """Compute statistics about the text data."""
    sentences = df['sentence'].tolist()
    
    # Character counts
    char_counts = [len(s) for s in sentences]
    
    # Word counts (split by space)
    word_counts = [len(s.split()) for s in sentences]
    
    # Unique characters
    all_chars = set(''.join(sentences))
    bangla_chars = set(c for c in all_chars if '\u0980' <= c <= '\u09FF')
    
    stats = {
        'total_samples': len(df),
        'total_characters': sum(char_counts),
        'total_words': sum(word_counts),
        'avg_chars_per_sample': np.mean(char_counts),
        'avg_words_per_sample': np.mean(word_counts),
        'min_chars': min(char_counts),
        'max_chars': max(char_counts),
        'min_words': min(word_counts),
        'max_words': max(word_counts),
        'unique_characters': len(all_chars),
        'unique_bangla_characters': len(bangla_chars),
        'bangla_chars': ''.join(sorted(bangla_chars)),
    }
    
    return stats


def split_dataset(
    df: pd.DataFrame,
    valid_ratio: float = 0.2,
    stratify_by_length: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    """
    Split dataset into train and valid sets.
    
    Args:
        df: Input DataFrame with 'id' and 'sentence' columns
        valid_ratio: Fraction of data for validation (0.1 = 10%)
        stratify_by_length: If True, stratify by sentence length
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with added 'split' column
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n_samples = len(df)
    n_valid = int(n_samples * valid_ratio)
    n_train = n_samples - n_valid
    
    if stratify_by_length:
        # Stratify by sentence length (short/medium/long)
        df = df.copy()
        char_counts = df['sentence'].str.len()
        
        # Create length bins
        df['_length_bin'] = pd.qcut(char_counts, q=3, labels=['short', 'medium', 'long'])
        
        # Sample from each bin proportionally
        valid_indices = []
        for bin_name in ['short', 'medium', 'long']:
            bin_indices = df[df['_length_bin'] == bin_name].index.tolist()
            n_bin_valid = max(1, int(len(bin_indices) * valid_ratio))
            valid_indices.extend(random.sample(bin_indices, n_bin_valid))
        
        df = df.drop(columns=['_length_bin'])
        
    else:
        # Random split
        all_indices = list(range(n_samples))
        random.shuffle(all_indices)
        valid_indices = all_indices[:n_valid]
    
    # Assign splits
    df = df.copy()
    df['split'] = 'train'
    df.loc[valid_indices, 'split'] = 'valid'
    
    return df


def filter_by_length(
    df: pd.DataFrame,
    min_chars: int = 1,
    max_chars: int = 500,
    min_words: int = 1,
    max_words: int = 100
) -> pd.DataFrame:
    """Filter samples by text length."""
    original_len = len(df)
    
    char_counts = df['sentence'].str.len()
    word_counts = df['sentence'].str.split().str.len()
    
    mask = (
        (char_counts >= min_chars) & 
        (char_counts <= max_chars) &
        (word_counts >= min_words) &
        (word_counts <= max_words)
    )
    
    filtered_df = df[mask].reset_index(drop=True)
    
    removed = original_len - len(filtered_df)
    if removed > 0:
        print(f"Filtered out {removed} samples based on length criteria")
    
    return filtered_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries (by ID or sentence)."""
    original_len = len(df)
    
    # Remove duplicate IDs
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    # Optionally remove duplicate sentences (uncomment if needed)
    # df = df.drop_duplicates(subset=['sentence'], keep='first')
    
    df = df.reset_index(drop=True)
    
    removed = original_len - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate entries")
    
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Basic text cleaning."""
    df = df.copy()
    
    # Strip whitespace
    df['sentence'] = df['sentence'].str.strip()
    
    # Normalize whitespace
    df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True)
    
    # Remove empty sentences
    df = df[df['sentence'].str.len() > 0]
    
    return df.reset_index(drop=True)


def print_stats(df: pd.DataFrame, stats: dict):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\n📊 Overall:")
    print(f"   Total samples: {stats['total_samples']:,}")
    print(f"   Total characters: {stats['total_characters']:,}")
    print(f"   Total words: {stats['total_words']:,}")
    
    print(f"\n📏 Per Sample:")
    print(f"   Avg characters: {stats['avg_chars_per_sample']:.1f}")
    print(f"   Avg words: {stats['avg_words_per_sample']:.1f}")
    print(f"   Char range: {stats['min_chars']} - {stats['max_chars']}")
    print(f"   Word range: {stats['min_words']} - {stats['max_words']}")
    
    print(f"\n🔤 Vocabulary:")
    print(f"   Unique characters: {stats['unique_characters']}")
    print(f"   Unique Bangla chars: {stats['unique_bangla_characters']}")
    
    if 'split' in df.columns:
        split_counts = df['split'].value_counts()
        print(f"\n📂 Split Distribution:")
        for split_name, count in split_counts.items():
            pct = count / len(df) * 100
            print(f"   {split_name}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)


def print_samples(df: pd.DataFrame, n: int = 5):
    """Print sample entries."""
    print(f"\n📝 Sample Entries (first {n}):")
    print("-" * 60)
    for idx, row in df.head(n).iterrows():
        split_str = f" [{row['split']}]" if 'split' in df.columns else ""
        print(f"ID: {row['id']}{split_str}")
        print(f"   {row['sentence'][:80]}{'...' if len(row['sentence']) > 80 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare and split dataset for Bangla ASR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic split
    python prepare_data.py --input raw_data.csv --output data/train.csv
    
    # With validation
    python prepare_data.py --input raw.csv --output data/train.csv --audio-dir data/train --validate
    
    # Custom split ratio
    python prepare_data.py --input raw.csv --output data/train.csv --valid-ratio 0.15
    
    # Stratified split
    python prepare_data.py --input raw.csv --output data/train.csv --stratify
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input CSV/JSON file with transcriptions')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output CSV path (e.g., data/train.csv)')
    parser.add_argument('--audio-dir', '-a', type=str, default=None,
                       help='Audio directory to validate files exist')
    parser.add_argument('--valid-ratio', '-v', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1 = 10%%)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate audio files exist')
    parser.add_argument('--stratify', action='store_true',
                       help='Stratify split by sentence length')
    parser.add_argument('--min-chars', type=int, default=1,
                       help='Minimum characters per sentence')
    parser.add_argument('--max-chars', type=int, default=500,
                       help='Maximum characters per sentence')
    parser.add_argument('--stats-output', type=str, default=None,
                       help='Save statistics to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading data from: {input_path}")
    
    # Load data
    df = load_input_data(input_path, audio_dir)
    print(f"   Loaded {len(df):,} entries")
    
    # Clean text
    df = clean_text(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Filter by length
    df = filter_by_length(
        df, 
        min_chars=args.min_chars,
        max_chars=args.max_chars
    )
    
    # Validate audio files
    if args.validate and audio_dir:
        print(f"\n🔍 Validating audio files in: {audio_dir}")
        df, missing = validate_audio_files(df, audio_dir)
        
        if missing:
            print(f"   ⚠️  Missing {len(missing)} audio files")
            if len(missing) <= 10:
                for mid in missing:
                    print(f"      - {mid}")
            else:
                print(f"      First 10: {missing[:10]}")
        else:
            print(f"   ✅ All audio files found")
    
    # Split dataset
    print(f"\n✂️  Splitting dataset (valid_ratio={args.valid_ratio})")
    df = split_dataset(
        df,
        valid_ratio=args.valid_ratio,
        stratify_by_length=args.stratify,
        seed=args.seed
    )
    
    # Compute statistics
    stats = compute_text_stats(df)
    
    # Print info
    if not args.quiet:
        print_stats(df, stats)
        print_samples(df)
    
    # Save output
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    # Save statistics
    if args.stats_output:
        stats_path = Path(args.stats_output)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📊 Statistics saved to: {stats_path}")
    
    # Summary
    train_count = len(df[df['split'] == 'train'])
    valid_count = len(df[df['split'] == 'valid'])
    print(f"\n✅ Done! Train: {train_count:,} | Valid: {valid_count:,}")


if __name__ == '__main__':
    main()
