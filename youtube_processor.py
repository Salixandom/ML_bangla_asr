#!/usr/bin/env python
"""
YouTube Audio Processor for ASR Training

Downloads audio from YouTube, fetches lyrics, and creates speech chunks
with aligned transcriptions for ASR training.

Features:
- Downloads audio from YouTube using yt-dlp
- Fetches lyrics from LRCLib (synced) or Genius API (plain)
- Chunks audio into 10-15 second segments
- Filters: KEEPS speech+music chunks, REMOVES music-only chunks
- Aligns lyrics to chunks for transcription
- Outputs manifest CSV compatible with the ASR pipeline

Requirements:
    pip install yt-dlp pydub webrtcvad-wheels librosa soundfile lyricsgenius requests

Usage:
    python youtube_processor.py --url "https://youtube.com/watch?v=..." --output ./youtube_data
    python youtube_processor.py --url "..." --genius-token YOUR_TOKEN --output ./youtube_data
"""

import argparse
import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import requests

import numpy as np
import soundfile as sf
import librosa


@dataclass
class LyricLine:
    """A single line of lyrics with timing."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds


@dataclass 
class AudioChunk:
    """Represents an audio chunk with metadata."""
    audio: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    has_speech: bool
    speech_ratio: float
    energy: float
    transcript: str = ""
    

class LyricsFetcher:
    """Fetch lyrics from multiple sources."""
    
    def __init__(self, genius_token: Optional[str] = None):
        self.genius_token = genius_token
    
    def fetch_synced_lyrics(self, title: str, artist: str = None) -> Optional[List[LyricLine]]:
        """
        Try to fetch time-synced lyrics (LRC format) from LRCLib.
        
        Args:
            title: Song title
            artist: Artist name
            
        Returns:
            List of LyricLine objects with timing, or None
        """
        try:
            # Try LRCLib API (free, no token needed, has synced lyrics)
            params = {'track_name': title}
            if artist:
                params['artist_name'] = artist
            
            response = requests.get(
                'https://lrclib.net/api/search',
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                if results:
                    # Get the first result with synced lyrics
                    for result in results:
                        if result.get('syncedLyrics'):
                            return self._parse_lrc(result['syncedLyrics'])
            
            print(f"   No synced lyrics found on LRCLib for: {title}")
            return None
            
        except Exception as e:
            print(f"   Error fetching synced lyrics: {e}")
            return None
    
    def _parse_lrc(self, lrc_text: str) -> List[LyricLine]:
        """Parse LRC format lyrics into LyricLine objects."""
        lines = []
        pattern = r'\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)'
        
        raw_lines = []
        for line in lrc_text.strip().split('\n'):
            match = re.match(pattern, line)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                centiseconds = int(match.group(3))
                
                # Convert to seconds
                if len(match.group(3)) == 2:
                    time_seconds = minutes * 60 + seconds + centiseconds / 100
                else:
                    time_seconds = minutes * 60 + seconds + centiseconds / 1000
                
                text = match.group(4).strip()
                if text:  # Skip empty lines
                    raw_lines.append((time_seconds, text))
        
        # Create LyricLine objects with end times
        for i, (start_time, text) in enumerate(raw_lines):
            if i < len(raw_lines) - 1:
                end_time = raw_lines[i + 1][0]
            else:
                end_time = start_time + 5.0  # Assume 5 seconds for last line
            
            lines.append(LyricLine(
                text=text,
                start_time=start_time,
                end_time=end_time
            ))
        
        return lines
    
    def fetch_plain_lyrics(self, title: str, artist: str = None) -> Optional[str]:
        """
        Fetch plain lyrics from Genius.
        
        Args:
            title: Song title
            artist: Artist name
            
        Returns:
            Plain lyrics text or None
        """
        if not self.genius_token:
            return None
        
        try:
            import lyricsgenius
            
            genius = lyricsgenius.Genius(
                self.genius_token,
                verbose=False,
                remove_section_headers=True
            )
            
            song = genius.search_song(title, artist)
            
            if song:
                print(f"   Found lyrics on Genius: {song.title} by {song.artist}")
                return song.lyrics
            
            return None
            
        except ImportError:
            print("   lyricsgenius not installed. Install with: pip install lyricsgenius")
            return None
        except Exception as e:
            print(f"   Error fetching from Genius: {e}")
            return None
    
    def fetch(self, title: str, artist: str = None, duration: float = None) -> Tuple[Optional[List[LyricLine]], Optional[str]]:
        """
        Fetch lyrics, trying synced first, then plain.
        
        Returns:
            Tuple of (synced_lyrics, plain_lyrics)
        """
        # Try synced lyrics first
        synced = self.fetch_synced_lyrics(title, artist)
        
        # Try plain lyrics
        plain = self.fetch_plain_lyrics(title, artist)
        
        # If we have plain but no synced, try to create approximate timing
        if plain and not synced and duration:
            synced = self._create_approximate_timing(plain, duration)
        
        return synced, plain
    
    def _create_approximate_timing(self, plain_lyrics: str, duration: float) -> List[LyricLine]:
        """Create approximate timing for plain lyrics based on duration."""
        # Clean and split lyrics into lines
        lines = [l.strip() for l in plain_lyrics.split('\n') if l.strip()]
        
        # Remove common non-lyric patterns
        lines = [l for l in lines if not l.startswith('[') and not 'Embed' in l and not 'Lyrics' in l]
        
        if not lines:
            return []
        
        # Distribute evenly across duration (leaving some buffer at start/end)
        usable_duration = duration * 0.85  # Use 85% of duration
        start_offset = duration * 0.075    # Start at 7.5%
        
        time_per_line = usable_duration / len(lines)
        
        result = []
        for i, text in enumerate(lines):
            start_time = start_offset + i * time_per_line
            end_time = start_time + time_per_line
            
            result.append(LyricLine(
                text=text,
                start_time=start_time,
                end_time=end_time
            ))
        
        return result


class YouTubeProcessor:
    """
    Process YouTube videos for ASR training.
    
    Downloads audio, detects speech segments, aligns lyrics, and creates training chunks.
    """
    
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        chunk_min_duration: float = 10.0,
        chunk_max_duration: float = 15.0,
        vad_aggressiveness: int = 2,
        min_speech_ratio: float = 0.2,  # Lower threshold to keep speech+music
        genius_token: Optional[str] = None
    ):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory for output files
            sample_rate: Target sample rate (16000 for wav2vec-BERT)
            chunk_min_duration: Minimum chunk length in seconds
            chunk_max_duration: Maximum chunk length in seconds
            vad_aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            min_speech_ratio: Minimum ratio of speech frames to keep chunk (0.2 = 20%)
            genius_token: Genius API token for lyrics (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.chunk_min_duration = chunk_min_duration
        self.chunk_max_duration = chunk_max_duration
        self.vad_aggressiveness = vad_aggressiveness
        self.min_speech_ratio = min_speech_ratio
        
        # Lyrics fetcher
        self.lyrics_fetcher = LyricsFetcher(genius_token)
        
        # Create subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.chunks_dir = self.output_dir / "chunks"
        self.audio_dir.mkdir(exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Initialize VAD
        self.vad = self._init_vad()
    
    def _init_vad(self):
        """Initialize Voice Activity Detection."""
        try:
            import webrtcvad
            return webrtcvad.Vad(self.vad_aggressiveness)
        except ImportError:
            print("Warning: webrtcvad not available, using energy-based VAD")
            return None
    
    def download_audio(self, url: str) -> Tuple[Path, Dict]:
        """
        Download audio from YouTube.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (audio_path, metadata_dict)
        """
        print(f"\n📥 Downloading audio from: {url}")
        
        # Output template
        output_template = str(self.audio_dir / "%(title)s.%(ext)s")
        
        # yt-dlp command for audio extraction
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "-o", output_template,
            "--print-json",  # Print metadata as JSON
            "--no-playlist",  # Don't download playlists
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse metadata from JSON output
            metadata = json.loads(result.stdout.strip().split('\n')[-1])
            
            # Find the downloaded file
            title = self._sanitize_filename(metadata.get('title', 'audio'))
            audio_path = self.audio_dir / f"{title}.wav"
            
            # If file doesn't exist with exact name, find it
            if not audio_path.exists():
                wav_files = list(self.audio_dir.glob("*.wav"))
                if wav_files:
                    audio_path = wav_files[-1]  # Most recent
            
            print(f"✅ Downloaded: {audio_path.name}")
            print(f"   Title: {metadata.get('title', 'Unknown')}")
            print(f"   Duration: {metadata.get('duration', 0)}s")
            
            return audio_path, metadata
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        return filename[:100]
    
    def load_and_preprocess(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio file."""
        print(f"\n🔄 Loading and preprocessing audio...")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        print(f"   Duration: {len(audio)/self.sample_rate:.2f}s at {self.sample_rate}Hz")
        
        return audio
    
    def detect_speech_frames(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect speech frames using VAD.
        
        Returns boolean array where True = speech detected.
        """
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        
        num_frames = len(audio) // frame_size
        speech_frames = np.zeros(num_frames, dtype=bool)
        
        if self.vad is not None:
            # Use webrtcvad
            audio_int16 = (audio * 32767).astype(np.int16)
            
            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio_int16[start:end].tobytes()
                
                try:
                    speech_frames[i] = self.vad.is_speech(frame, self.sample_rate)
                except:
                    # Fallback to energy-based
                    frame_audio = audio[start:end]
                    energy = np.sqrt(np.mean(frame_audio ** 2))
                    speech_frames[i] = energy > 0.02
        else:
            # Energy-based VAD fallback
            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                frame_audio = audio[start:end]
                energy = np.sqrt(np.mean(frame_audio ** 2))
                speech_frames[i] = energy > 0.02
        
        return speech_frames
    
    def get_lyrics_for_timerange(
        self, 
        lyrics: List[LyricLine], 
        start_time: float, 
        end_time: float
    ) -> str:
        """
        Get lyrics text for a specific time range.
        
        Args:
            lyrics: List of LyricLine objects
            start_time: Chunk start time
            end_time: Chunk end time
            
        Returns:
            Combined lyrics text for the time range
        """
        if not lyrics:
            return ""
        
        matching_lines = []
        
        for line in lyrics:
            # Check if lyric line overlaps with chunk time range
            # Line overlaps if: line.start < chunk.end AND line.end > chunk.start
            if line.start_time < end_time and line.end_time > start_time:
                matching_lines.append(line.text)
        
        # Join lines with space
        return ' '.join(matching_lines)
    
    def create_chunks(
        self, 
        audio: np.ndarray, 
        lyrics: Optional[List[LyricLine]] = None
    ) -> List[AudioChunk]:
        """
        Create chunks from audio, filtering for speech content.
        
        KEEPS: Chunks with speech (even with music/noise)
        REMOVES: Chunks with ONLY music/noise (no speech)
        
        Args:
            audio: Full audio array
            lyrics: Optional list of LyricLine for alignment
            
        Returns:
            List of AudioChunk objects with speech content
        """
        print("\n🎵 Creating chunks...")
        
        chunks = []
        total_duration = len(audio) / self.sample_rate
        
        # Detect speech frames for full audio
        print("   Running voice activity detection...")
        speech_frames = self.detect_speech_frames(audio)
        frame_duration = 0.03  # 30ms frames
        
        # Create fixed-size chunks
        chunk_duration = (self.chunk_min_duration + self.chunk_max_duration) / 2  # ~12.5s
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        position = 0
        chunk_id = 0
        kept_chunks = 0
        discarded_chunks = 0
        
        while position + chunk_samples <= len(audio):
            # Extract chunk
            chunk_audio = audio[position:position + chunk_samples]
            
            # Get corresponding speech frames
            frame_start = int(position / self.sample_rate / frame_duration)
            frame_end = int((position + chunk_samples) / self.sample_rate / frame_duration)
            frame_end = min(frame_end, len(speech_frames))
            chunk_speech_frames = speech_frames[frame_start:frame_end]
            
            # Calculate speech ratio
            speech_ratio = np.mean(chunk_speech_frames) if len(chunk_speech_frames) > 0 else 0
            
            # Compute energy
            energy = np.sqrt(np.mean(chunk_audio ** 2))
            
            start_time = position / self.sample_rate
            end_time = (position + chunk_samples) / self.sample_rate
            
            # Decision: Keep if has ANY significant speech (speech_ratio >= min_speech_ratio)
            # This keeps speech+music chunks but removes pure music/instrumental
            has_speech = speech_ratio >= self.min_speech_ratio
            
            # Get aligned lyrics for this chunk
            transcript = ""
            if lyrics:
                transcript = self.get_lyrics_for_timerange(lyrics, start_time, end_time)
            
            if has_speech:
                chunk = AudioChunk(
                    audio=chunk_audio,
                    start_time=start_time,
                    end_time=end_time,
                    sample_rate=self.sample_rate,
                    has_speech=True,
                    speech_ratio=speech_ratio,
                    energy=energy,
                    transcript=transcript
                )
                chunks.append(chunk)
                kept_chunks += 1
            else:
                discarded_chunks += 1
            
            chunk_id += 1
            position += chunk_samples  # Non-overlapping chunks
        
        print(f"   Total chunks analyzed: {chunk_id}")
        print(f"   ✅ Kept (has speech): {kept_chunks}")
        print(f"   ❌ Discarded (music only): {discarded_chunks}")
        
        return chunks
    
    def save_chunks(self, chunks: List[AudioChunk], base_name: str) -> List[Dict]:
        """
        Save chunks to disk and create manifest entries.
        
        Args:
            chunks: List of AudioChunk objects
            base_name: Base name for chunk files
            
        Returns:
            List of manifest entries
        """
        manifest_entries = []
        
        for i, chunk in enumerate(chunks):
            # Generate filename
            filename = f"{base_name}_chunk_{i:04d}.wav"
            filepath = self.chunks_dir / filename
            
            # Save audio
            sf.write(filepath, chunk.audio, chunk.sample_rate)
            
            # Create manifest entry with transcript
            entry = {
                'id': f"{base_name}_{i:04d}",
                'audio_path': str(filepath.absolute()),
                'duration': chunk.end_time - chunk.start_time,
                'start_time': chunk.start_time,
                'end_time': chunk.end_time,
                'speech_ratio': round(chunk.speech_ratio, 3),
                'sentence': chunk.transcript,  # Aligned lyrics as transcript
                'split': 'train'
            }
            manifest_entries.append(entry)
        
        print(f"\n💾 Saved {len(chunks)} chunks to {self.chunks_dir}")
        
        return manifest_entries
    
    def save_manifest(self, entries: List[Dict], filename: str = "manifest.csv"):
        """Save manifest CSV file."""
        import csv
        
        manifest_path = self.output_dir / filename
        
        fieldnames = ['id', 'audio_path', 'duration', 'start_time', 'end_time', 
                      'speech_ratio', 'sentence', 'split']
        
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
        
        print(f"📋 Manifest saved to {manifest_path}")
        return manifest_path
    
    def save_lyrics(self, lyrics: List[LyricLine], plain_lyrics: str, filename: str = "lyrics.json"):
        """Save lyrics to JSON file."""
        lyrics_path = self.output_dir / filename
        
        data = {
            'synced_lyrics': [
                {
                    'text': l.text,
                    'start_time': l.start_time,
                    'end_time': l.end_time
                }
                for l in lyrics
            ] if lyrics else [],
            'plain_lyrics': plain_lyrics or ""
        }
        
        with open(lyrics_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"🎤 Lyrics saved to {lyrics_path}")
        return lyrics_path
    
    def process(self, url: str) -> Dict:
        """
        Full processing pipeline.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with processing results
        """
        print("\n" + "="*60)
        print("🎬 YouTube Audio Processor for ASR Training")
        print("="*60)
        
        # Step 1: Download audio
        audio_path, metadata = self.download_audio(url)
        duration = metadata.get('duration', 0)
        
        # Step 2: Fetch lyrics
        print("\n🔍 Fetching lyrics...")
        title = metadata.get('title', '')
        
        # Try to extract artist from title (common format: "Artist - Song")
        artist = None
        song_title = title
        if ' - ' in title:
            parts = title.split(' - ', 1)
            artist = parts[0].strip()
            song_title = parts[1].strip()
        
        synced_lyrics, plain_lyrics = self.lyrics_fetcher.fetch(
            song_title, 
            artist, 
            duration=duration
        )
        
        if synced_lyrics:
            print(f"   ✅ Found {len(synced_lyrics)} synced lyric lines")
        elif plain_lyrics:
            print(f"   ⚠️ Only plain lyrics found (using approximate timing)")
            # Create approximate timing
            synced_lyrics = self.lyrics_fetcher._create_approximate_timing(plain_lyrics, duration)
            print(f"   📝 Created {len(synced_lyrics)} timed lines")
        else:
            print(f"   ❌ No lyrics found - chunks will have empty transcripts")
        
        # Step 3: Load and preprocess
        audio = self.load_and_preprocess(audio_path)
        
        # Step 4: Create chunks with speech detection and lyrics alignment
        chunks = self.create_chunks(audio, synced_lyrics)
        
        if not chunks:
            print("\n⚠️ Warning: No speech chunks detected!")
            return {
                'success': False,
                'message': 'No speech content detected',
                'metadata': metadata
            }
        
        # Step 5: Save chunks
        base_name = self._sanitize_filename(metadata.get('title', 'audio'))
        manifest_entries = self.save_chunks(chunks, base_name)
        
        # Step 6: Save manifest
        manifest_path = self.save_manifest(manifest_entries)
        
        # Step 7: Save lyrics
        if synced_lyrics or plain_lyrics:
            self.save_lyrics(synced_lyrics or [], plain_lyrics)
        
        # Step 8: Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'url': url,
                'title': metadata.get('title'),
                'duration': duration,
                'total_chunks': len(chunks),
                'chunks_with_lyrics': sum(1 for e in manifest_entries if e['sentence']),
                'has_synced_lyrics': synced_lyrics is not None and len(synced_lyrics) > 0,
                'has_plain_lyrics': plain_lyrics is not None,
            }, f, indent=2, ensure_ascii=False)
        
        # Summary
        chunks_with_lyrics = sum(1 for e in manifest_entries if e['sentence'])
        total_speech_duration = sum(c.end_time - c.start_time for c in chunks)
        
        print("\n" + "="*60)
        print("✅ Processing Complete!")
        print("="*60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎵 Audio chunks: {len(chunks)}")
        print(f"📝 Chunks with lyrics: {chunks_with_lyrics}")
        print(f"⏱️ Total speech duration: {total_speech_duration:.1f}s")
        print(f"📋 Manifest: {manifest_path}")
        
        # Show sample entries
        print("\n📄 Sample manifest entries:")
        for entry in manifest_entries[:3]:
            print(f"   [{entry['id']}] {entry['start_time']:.1f}s-{entry['end_time']:.1f}s")
            print(f"      Speech: {entry['speech_ratio']*100:.0f}%")
            print(f"      Text: {entry['sentence'][:60]}..." if len(entry['sentence']) > 60 else f"      Text: {entry['sentence']}")
        
        print(f"\n🚀 To use with ASR pipeline:")
        print(f"   python run.py train --model bangla --manifest {manifest_path}")
        
        return {
            'success': True,
            'output_dir': str(self.output_dir),
            'manifest_path': str(manifest_path),
            'num_chunks': len(chunks),
            'chunks_with_lyrics': chunks_with_lyrics,
            'total_duration': total_speech_duration,
            'metadata': metadata
        }


def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube audio and create ASR training chunks with lyrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (tries LRCLib for synced lyrics - no API key needed)
  python youtube_processor.py --url "https://youtube.com/watch?v=..." --output ./youtube_data
  
  # With Genius API for lyrics fallback
  python youtube_processor.py --url "..." --genius-token YOUR_TOKEN --output ./youtube_data
  
  # Adjust speech detection threshold
  python youtube_processor.py --url "..." --min-speech-ratio 0.15  # Keep more chunks (more noisy)
  python youtube_processor.py --url "..." --min-speech-ratio 0.4   # Stricter filtering

How it works:
  1. Downloads audio from YouTube using yt-dlp
  2. Fetches synced lyrics from LRCLib (or plain from Genius)
  3. Splits audio into 10-15 second chunks
  4. Uses Voice Activity Detection to identify speech
  5. KEEPS chunks with speech (even with background music) - good for noisy ASR training
  6. REMOVES chunks with only music/instrumentals - no speech to learn from
  7. Aligns lyrics to chunks based on timestamps
  8. Outputs manifest.csv compatible with ASR training pipeline

Output:
  ./youtube_data/
  ├── audio/           # Original downloaded audio
  ├── chunks/          # Speech chunks (10-15s each)
  ├── manifest.csv     # Training manifest with transcripts
  ├── lyrics.json      # Full lyrics data
  └── metadata.json    # Video information
        """
    )
    
    parser.add_argument('--url', '-u', type=str, required=True,
                       help='YouTube video URL')
    parser.add_argument('--output', '-o', type=str, default='./youtube_data',
                       help='Output directory')
    parser.add_argument('--genius-token', type=str, default=None,
                       help='Genius API token for lyrics fallback (optional)')
    parser.add_argument('--min-chunk', type=float, default=10.0,
                       help='Minimum chunk duration in seconds')
    parser.add_argument('--max-chunk', type=float, default=15.0,
                       help='Maximum chunk duration in seconds')
    parser.add_argument('--min-speech-ratio', type=float, default=0.2,
                       help='Minimum speech ratio to keep chunk (0.2 = 20%% speech)')
    parser.add_argument('--vad-aggressiveness', type=int, default=2, choices=[0,1,2,3],
                       help='VAD aggressiveness (0=lenient, 3=strict)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = YouTubeProcessor(
        output_dir=args.output,
        chunk_min_duration=args.min_chunk,
        chunk_max_duration=args.max_chunk,
        min_speech_ratio=args.min_speech_ratio,
        vad_aggressiveness=args.vad_aggressiveness,
        genius_token=args.genius_token
    )
    
    # Process
    result = processor.process(url=args.url)
    
    if result['success']:
        print(f"\n🎉 Successfully processed {result['num_chunks']} chunks")
        print(f"   {result['chunks_with_lyrics']} chunks have aligned lyrics")
    else:
        print(f"\n❌ Processing failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
