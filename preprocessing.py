"""
Audio Preprocessing Module for Bangla ASR Pipeline

Steps:
1. MP3 → PCM waveform decoding
2. Resampling to 16 kHz
3. Loudness normalization (LUFS)
4. Amplitude normalization [-1, 1]
5. Voice Activity Detection
6. Speech segmentation
7. Chunking

All steps are deterministic and applied to ALL splits.

Performance features:
- Multi-threading for parallel file processing
- GPU acceleration for resampling (if available)
- Batch processing support
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import pyloudnorm as pyln
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings

# Optional librosa (fallback)
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    print("WARNING: webrtcvad not installed!")
    print("For noisy audio, install it with: pip install webrtcvad-wheels")
    print("Falling back to energy-based VAD (less accurate for noisy audio).")

from config import AudioConfig, VADConfig


# Named tuple for speech segments
SpeechSegment = namedtuple('SpeechSegment', ['start', 'end', 'audio'])


@dataclass
class ProcessedAudio:
    """Container for processed audio data"""
    waveform: np.ndarray
    sample_rate: int
    segments: List[SpeechSegment]
    chunks: List[Tuple[np.ndarray, float, float]]  # (audio, start_time, end_time)
    original_duration: float
    processed_duration: float


class GPUResampler:
    """GPU-accelerated resampling using torchaudio."""
    
    def __init__(self, orig_sr: int, target_sr: int, device: str = 'cuda'):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Cache resamplers for common sample rates
        self._resamplers = {}
    
    def _get_resampler(self, orig_sr: int) -> T.Resample:
        """Get or create a resampler for the given sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sr,
                resampling_method='sinc_interp_kaiser',
                dtype=torch.float32
            ).to(self.device)
        return self._resamplers[orig_sr]
    
    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio using GPU if available."""
        if orig_sr == self.target_sr:
            return waveform
        
        # Convert to tensor
        tensor = torch.from_numpy(waveform).float()
        
        # Add batch dimension if needed
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        # Move to device and resample
        tensor = tensor.to(self.device)
        resampler = self._get_resampler(orig_sr)
        
        with torch.no_grad():
            resampled = resampler(tensor)
        
        # Move back to CPU and convert to numpy
        result = resampled.squeeze(0).cpu().numpy()
        
        return result.astype(np.float32)


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for wav2vec-BERT 2.0
    
    Features:
    - GPU-accelerated resampling
    - Efficient audio loading with torchaudio
    """
    
    def __init__(
        self, 
        audio_config: AudioConfig, 
        vad_config: VADConfig,
        use_gpu: bool = True
    ):
        self.audio_config = audio_config
        self.vad_config = vad_config
        self.meter = pyln.Meter(audio_config.sample_rate)
        
        # GPU resampler
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.gpu_resampler = GPUResampler(
            orig_sr=audio_config.sample_rate,
            target_sr=audio_config.sample_rate,
            device=self.device
        )
        
        if self.use_gpu:
            print(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        
        if HAS_WEBRTCVAD:
            self.vad = webrtcvad.Vad(vad_config.aggressiveness)
        else:
            self.vad = None
    
    def process_file(self, audio_path: Union[str, Path]) -> ProcessedAudio:
        """
        Complete preprocessing pipeline for a single audio file.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            
        Returns:
            ProcessedAudio object with all processed data
        """
        audio_path = Path(audio_path)
        
        # Step 1: Load and decode audio (using torchaudio for speed)
        waveform, original_sr = self._load_audio(audio_path)
        original_duration = len(waveform) / original_sr
        
        # Step 2: Resample to 16 kHz (GPU accelerated)
        if original_sr != self.audio_config.sample_rate:
            waveform = self.gpu_resampler.resample(waveform, original_sr)
        
        # Step 3: Loudness normalization
        waveform = self._normalize_loudness(waveform)
        
        # Step 4: Amplitude normalization
        waveform = self._normalize_amplitude(waveform)
        
        # Step 5 & 6: VAD and speech segmentation
        segments = self._extract_speech_segments(waveform)
        
        # Step 7: Chunking
        chunks = self._chunk_segments(segments)
        
        processed_duration = sum(len(c[0]) for c in chunks) / self.audio_config.sample_rate
        
        return ProcessedAudio(
            waveform=waveform,
            sample_rate=self.audio_config.sample_rate,
            segments=segments,
            chunks=chunks,
            original_duration=original_duration,
            processed_duration=processed_duration
        )
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Step 1: Decode audio file to mono float32 waveform.
        Uses torchaudio for WAV/FLAC, librosa for MP3 (more reliable).
        """
        suffix = audio_path.suffix.lower()
        
        # Use librosa for MP3 (torchaudio has FFmpeg issues on some systems)
        if suffix == '.mp3':
            if HAS_LIBROSA:
                waveform, sr = librosa.load(audio_path, sr=None, mono=True)
                return waveform.astype(np.float32), sr
            else:
                raise RuntimeError(f"librosa required for MP3 files: {audio_path}")
        
        # Try torchaudio for WAV/FLAC (faster)
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0).numpy()
            return waveform.astype(np.float32), sr
            
        except Exception as e:
            # Fallback to librosa
            if HAS_LIBROSA:
                waveform, sr = librosa.load(audio_path, sr=None, mono=True)
                return waveform.astype(np.float32), sr
            else:
                raise RuntimeError(f"Could not load {audio_path}: {e}")
    
    def _normalize_loudness(self, waveform: np.ndarray) -> np.ndarray:
        """
        Step 3: Normalize loudness to target LUFS (-23 LUFS default).
        Removes recording-device loudness bias.
        """
        # Measure current loudness
        current_loudness = self.meter.integrated_loudness(waveform)
        
        # Handle silent or very quiet audio
        if np.isinf(current_loudness) or current_loudness < -70:
            return waveform
        
        # Normalize to target loudness
        normalized = pyln.normalize.loudness(
            waveform, 
            current_loudness, 
            self.audio_config.target_lufs
        )
        return normalized.astype(np.float32)
    
    def _normalize_amplitude(self, waveform: np.ndarray) -> np.ndarray:
        """
        Step 4: Normalize waveform amplitude to [-1, 1].
        Prevents numerical instability.
        """
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform.astype(np.float32)
    
    def _extract_speech_segments(self, waveform: np.ndarray) -> List[SpeechSegment]:
        """
        Steps 5 & 6: VAD + speech segment extraction.
        """
        if self.vad is not None:
            return self._webrtc_vad_segments(waveform)
        else:
            return self._energy_vad_segments(waveform)
    
    def _webrtc_vad_segments(self, waveform: np.ndarray) -> List[SpeechSegment]:
        """
        WebRTC VAD-based speech segmentation.
        """
        sr = self.audio_config.sample_rate
        frame_duration_ms = self.vad_config.frame_duration_ms
        frame_size = int(sr * frame_duration_ms / 1000)
        
        # Convert to 16-bit PCM for webrtcvad
        waveform_int16 = (waveform * 32767).astype(np.int16)
        
        # Get frame-level speech/non-speech labels
        frames = []
        for i in range(0, len(waveform_int16) - frame_size, frame_size):
            frame = waveform_int16[i:i + frame_size].tobytes()
            try:
                is_speech = self.vad.is_speech(frame, sr)
            except:
                is_speech = False
            frames.append((i, is_speech))
        
        # Merge frames into segments with padding
        segments = self._merge_speech_frames(frames, waveform, frame_size)
        
        return segments
    
    def _energy_vad_segments(self, waveform: np.ndarray) -> List[SpeechSegment]:
        """
        Fallback energy-based VAD when webrtcvad is not available.
        """
        sr = self.audio_config.sample_rate
        frame_duration_ms = self.vad_config.frame_duration_ms
        frame_size = int(sr * frame_duration_ms / 1000)
        
        # Compute frame energies
        energies = []
        for i in range(0, len(waveform) - frame_size, frame_size):
            frame = waveform[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append((i, energy))
        
        if not energies:
            return []
        
        # Adaptive threshold
        energy_values = [e[1] for e in energies]
        threshold = np.percentile(energy_values, 30)  # Bottom 30% is silence
        
        # Label frames
        frames = [(pos, energy > threshold) for pos, energy in energies]
        
        # Merge frames into segments
        segments = self._merge_speech_frames(frames, waveform, frame_size)
        
        return segments
    
    def _merge_speech_frames(
        self, 
        frames: List[Tuple[int, bool]], 
        waveform: np.ndarray,
        frame_size: int
    ) -> List[SpeechSegment]:
        """
        Merge individual speech frames into continuous segments.
        """
        sr = self.audio_config.sample_rate
        padding_samples = int(self.vad_config.padding_duration_ms * sr / 1000)
        min_speech_samples = int(self.vad_config.min_speech_duration_ms * sr / 1000)
        merge_threshold_samples = int(self.audio_config.merge_threshold * sr)
        min_segment_samples = int(self.audio_config.min_segment_duration * sr)
        
        if not frames:
            return []
        
        # Find speech regions
        speech_regions = []
        in_speech = False
        speech_start = 0
        
        for i, (pos, is_speech) in enumerate(frames):
            if is_speech and not in_speech:
                speech_start = pos
                in_speech = True
            elif not is_speech and in_speech:
                speech_end = pos + frame_size
                if speech_end - speech_start >= min_speech_samples:
                    speech_regions.append((speech_start, speech_end))
                in_speech = False
        
        # Handle case where speech continues to the end
        if in_speech:
            speech_end = len(waveform)
            if speech_end - speech_start >= min_speech_samples:
                speech_regions.append((speech_start, speech_end))
        
        if not speech_regions:
            return []
        
        # Merge nearby segments
        merged_regions = [speech_regions[0]]
        for start, end in speech_regions[1:]:
            prev_start, prev_end = merged_regions[-1]
            if start - prev_end <= merge_threshold_samples:
                merged_regions[-1] = (prev_start, end)
            else:
                merged_regions.append((start, end))
        
        # Create segments with padding
        segments = []
        for start, end in merged_regions:
            # Add padding
            padded_start = max(0, start - padding_samples)
            padded_end = min(len(waveform), end + padding_samples)
            
            # Skip very short segments
            if padded_end - padded_start < min_segment_samples:
                continue
            
            segment_audio = waveform[padded_start:padded_end]
            segments.append(SpeechSegment(
                start=padded_start / sr,
                end=padded_end / sr,
                audio=segment_audio
            ))
        
        return segments
    
    def _chunk_segments(
        self, 
        segments: List[SpeechSegment]
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Step 7: Chunk speech segments into fixed-length windows.
        Handles long utterances by splitting with overlap.
        """
        sr = self.audio_config.sample_rate
        min_samples = int(self.audio_config.chunk_min_duration * sr)
        max_samples = int(self.audio_config.chunk_max_duration * sr)
        overlap_samples = int(self.audio_config.chunk_overlap * sr)
        
        chunks = []
        
        for segment in segments:
            audio = segment.audio
            segment_start = segment.start
            
            if len(audio) <= max_samples:
                # Segment fits in one chunk
                if len(audio) >= min_samples:
                    chunks.append((audio, segment_start, segment.end))
            else:
                # Split long segment into overlapping chunks
                pos = 0
                while pos < len(audio):
                    chunk_end = min(pos + max_samples, len(audio))
                    chunk_audio = audio[pos:chunk_end]
                    
                    if len(chunk_audio) >= min_samples:
                        chunk_start_time = segment_start + pos / sr
                        chunk_end_time = segment_start + chunk_end / sr
                        chunks.append((chunk_audio, chunk_start_time, chunk_end_time))
                    
                    # Move position with overlap
                    pos += max_samples - overlap_samples
                    
                    # Avoid tiny final chunks
                    if len(audio) - pos < min_samples and pos < len(audio):
                        break
        
        return chunks
    
    def process_for_model(
        self, 
        audio_path: Union[str, Path],
        return_chunks: bool = True
    ) -> Dict:
        """
        Process audio and return format ready for wav2vec-BERT 2.0.
        
        Returns:
            Dictionary with 'input_values' (raw waveform) ready for model
        """
        processed = self.process_file(audio_path)
        
        if return_chunks and processed.chunks:
            # Return list of chunk tensors
            return {
                'input_values': [torch.from_numpy(c[0]) for c, _, _ in processed.chunks],
                'chunk_times': [(start, end) for _, start, end in processed.chunks],
                'sample_rate': processed.sample_rate
            }
        else:
            # Return full processed waveform
            return {
                'input_values': torch.from_numpy(processed.waveform),
                'sample_rate': processed.sample_rate
            }


class TextPreprocessor:
    """
    Light text normalization for Bangla transcripts.
    
    Does:
    - Unicode normalization
    - Remove illegal characters
    - Standardize whitespace
    
    Does NOT:
    - Correct grammar
    - Change wording
    """
    
    # Bangla Unicode range
    BANGLA_RANGE = (0x0980, 0x09FF)
    
    # Additional allowed characters
    ALLOWED_CHARS = set(' \t\n।,?!.-০১২৩৪৫৬৭৮৯')
    
    def __init__(self):
        import unicodedata
        self.unicodedata = unicodedata
    
    def normalize(self, text: str) -> str:
        """
        Apply light normalization to Bangla text.
        """
        # Unicode NFC normalization
        text = self.unicodedata.normalize('NFC', text)
        
        # Filter characters
        filtered_chars = []
        for char in text:
            code_point = ord(char)
            # Keep Bangla characters
            if self.BANGLA_RANGE[0] <= code_point <= self.BANGLA_RANGE[1]:
                filtered_chars.append(char)
            # Keep allowed punctuation and whitespace
            elif char in self.ALLOWED_CHARS:
                filtered_chars.append(char)
            # Replace other characters with space
            else:
                filtered_chars.append(' ')
        
        text = ''.join(filtered_chars)
        
        # Standardize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def is_valid_bangla(self, text: str) -> bool:
        """Check if text contains valid Bangla characters."""
        bangla_chars = sum(
            1 for c in text 
            if self.BANGLA_RANGE[0] <= ord(c) <= self.BANGLA_RANGE[1]
        )
        return bangla_chars > 0


def _process_single_file(args):
    """
    Worker function for parallel processing.
    Returns tuple: (success, result_or_error)
    """
    row, audio_dir, output_dir, audio_config, vad_config, sample_rate = args
    
    audio_id = row['id']
    sentence = row['sentence']
    split = row['split']
    
    # Find audio file
    audio_path = None
    for ext in ['.mp3', '.wav', '.flac']:
        candidate = Path(audio_dir) / f"{audio_id}{ext}"
        if candidate.exists():
            audio_path = candidate
            break
    
    if audio_path is None:
        return (False, f"Audio not found: {audio_id}")
    
    try:
        # Create preprocessor (per-worker to avoid shared state issues)
        preprocessor = AudioPreprocessor(audio_config, vad_config, use_gpu=False)
        text_preprocessor = TextPreprocessor()
        
        # Process audio
        processed = preprocessor.process_file(audio_path)
        
        # Process text
        normalized_text = text_preprocessor.normalize(sentence)
        
        # Save processed audio chunks
        chunk_records = []
        output_audio_dir = Path(output_dir) / 'audio'
        
        for i, (chunk_audio, start_time, end_time) in enumerate(processed.chunks):
            chunk_filename = f"{audio_id}_chunk{i:03d}.wav"
            chunk_path = output_audio_dir / chunk_filename
            
            sf.write(
                chunk_path,
                chunk_audio,
                sample_rate,
                subtype='FLOAT'
            )
            
            chunk_records.append({
                'id': f"{audio_id}_chunk{i:03d}",
                'original_id': audio_id,
                'audio_path': str(chunk_path),
                'sentence': normalized_text,
                'split': split,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
        
        return (True, chunk_records)
    
    except Exception as e:
        return (False, f"Error processing {audio_id}: {str(e)}")


def preprocess_dataset(
    audio_dir: Path,
    csv_path: Path,
    output_dir: Path,
    audio_config: Optional[AudioConfig] = None,
    vad_config: Optional[VADConfig] = None,
    num_workers: int = None,
    use_gpu: bool = True
):
    """
    Batch preprocess entire dataset with parallel processing.
    
    Args:
        audio_dir: Directory containing audio files
        csv_path: Path to CSV with id, sentence, split columns
        output_dir: Directory for processed outputs
        audio_config: Audio preprocessing config
        vad_config: VAD config
        num_workers: Number of parallel workers (default: CPU count)
        use_gpu: Use GPU for resampling (only for single-threaded)
    """
    import pandas as pd
    from tqdm import tqdm
    
    audio_config = audio_config or AudioConfig()
    vad_config = vad_config or VADConfig()
    
    # Default workers = CPU count, but cap at 8 to avoid memory issues
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / 'audio').mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(df)} audio files...")
    print(f"Using {num_workers} workers")
    print(f"Output directory: {output_dir}")
    
    # Prepare arguments for workers
    worker_args = [
        (row.to_dict(), audio_dir, output_dir, audio_config, vad_config, audio_config.sample_rate)
        for _, row in df.iterrows()
    ]
    
    # Process files in parallel
    processed_records = []
    errors = []
    
    if num_workers > 1:
        # Multi-process for CPU-bound VAD operations
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_file, args): args[0]['id'] 
                      for args in worker_args}
            
            with tqdm(total=len(futures), desc="Processing") as pbar:
                for future in as_completed(futures):
                    audio_id = futures[future]
                    try:
                        success, result = future.result()
                        if success:
                            processed_records.extend(result)
                        else:
                            errors.append(result)
                    except Exception as e:
                        errors.append(f"Error with {audio_id}: {str(e)}")
                    pbar.update(1)
    else:
        # Single-threaded with GPU support
        preprocessor = AudioPreprocessor(audio_config, vad_config, use_gpu=use_gpu)
        text_preprocessor = TextPreprocessor()
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            audio_id = row['id']
            sentence = row['sentence']
            split = row['split']
            
            # Find audio file
            audio_path = None
            for ext in ['.mp3', '.wav', '.flac']:
                candidate = Path(audio_dir) / f"{audio_id}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
            
            if audio_path is None:
                errors.append(f"Audio not found: {audio_id}")
                continue
            
            try:
                processed = preprocessor.process_file(audio_path)
                normalized_text = text_preprocessor.normalize(sentence)
                
                for i, (chunk_audio, start_time, end_time) in enumerate(processed.chunks):
                    chunk_filename = f"{audio_id}_chunk{i:03d}.wav"
                    chunk_path = output_dir / 'audio' / chunk_filename
                    
                    sf.write(chunk_path, chunk_audio, audio_config.sample_rate, subtype='FLOAT')
                    
                    processed_records.append({
                        'id': f"{audio_id}_chunk{i:03d}",
                        'original_id': audio_id,
                        'audio_path': str(chunk_path),
                        'sentence': normalized_text,
                        'split': split,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
            except Exception as e:
                errors.append(f"Error processing {audio_id}: {str(e)}")
    
    # Save processed manifest
    processed_df = pd.DataFrame(processed_records)
    processed_df.to_csv(output_dir / 'manifest.csv', index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {len(processed_records)} chunks from {len(df) - len(errors)} files")
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"   - {err}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")
    print(f"📁 Manifest saved to: {output_dir / 'manifest.csv'}")
    print(f"{'='*60}")
    
    return processed_df


if __name__ == "__main__":
    # Example usage
    from config import get_config
    
    config = get_config()
    
    # Test single file processing
    preprocessor = AudioPreprocessor(config.audio, config.vad, use_gpu=True)
    
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # If you have a test file:
    # result = preprocessor.process_file("test_audio.mp3")
    # print(f"Original duration: {result.original_duration:.2f}s")
    # print(f"Processed duration: {result.processed_duration:.2f}s")
    # print(f"Number of chunks: {len(result.chunks)}")
