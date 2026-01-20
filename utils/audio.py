"""
Audio utilities for recording from microphone and playing back audio.
Cross-platform implementation using sounddevice.
"""

import numpy as np
import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice not installed. Run: pip install sounddevice")


@dataclass
class AudioConfig:
    """Audio configuration settings."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 1024
    silence_threshold: float = 0.01
    silence_duration: float = 1.5  # Seconds of silence to stop recording
    max_recording_duration: float = 30.0  # Maximum recording length


class AudioRecorder:
    """
    Records audio from the microphone.
    Supports manual start/stop and automatic silence detection.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        if sd is None:
            raise ImportError("sounddevice is required. Run: pip install sounddevice")
        
        self.config = config or AudioConfig()
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_recording = False
        self._stream: Optional[sd.InputStream] = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        self._audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start recording audio."""
        self._audio_queue = queue.Queue()
        self._is_recording = True
        
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the audio data."""
        self._is_recording = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Collect all audio data
        chunks = []
        while not self._audio_queue.empty():
            chunks.append(self._audio_queue.get())
        
        if chunks:
            return np.concatenate(chunks, axis=0).flatten()
        return np.array([], dtype=np.float32)
    
    def record_until_silence(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Record audio until silence is detected.
        
        Args:
            on_speech_start: Callback when speech starts
            on_speech_end: Callback when speech ends
            
        Returns:
            Recorded audio as numpy array
        """
        self.start_recording()
        
        chunks = []
        silence_start = None
        speech_detected = False
        start_time = time.time()
        
        try:
            while self._is_recording:
                # Check max duration
                if time.time() - start_time > self.config.max_recording_duration:
                    break
                
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                    chunks.append(chunk)
                    
                    # Check for silence
                    rms = np.sqrt(np.mean(chunk**2))
                    
                    if rms > self.config.silence_threshold:
                        # Speech detected
                        if not speech_detected:
                            speech_detected = True
                            if on_speech_start:
                                on_speech_start()
                        silence_start = None
                    else:
                        # Silence detected
                        if speech_detected:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self.config.silence_duration:
                                # Enough silence, stop recording
                                if on_speech_end:
                                    on_speech_end()
                                break
                
                except queue.Empty:
                    continue
        
        finally:
            self.stop_recording()
        
        if chunks:
            return np.concatenate(chunks, axis=0).flatten()
        return np.array([], dtype=np.float32)


class AudioPlayer:
    """
    Plays audio through speakers.
    Supports various audio formats.
    """
    
    def __init__(self, sample_rate: int = 22050):
        if sd is None:
            raise ImportError("sounddevice is required. Run: pip install sounddevice")
        self.sample_rate = sample_rate
    
    def play(self, audio_data: np.ndarray, sample_rate: Optional[int] = None, blocking: bool = True):
        """
        Play audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate (uses default if not specified)
            blocking: Wait for playback to finish
        """
        rate = sample_rate or self.sample_rate
        sd.play(audio_data, rate)
        if blocking:
            sd.wait()
    
    def play_file(self, filepath: str, blocking: bool = True):
        """
        Play audio from a file.
        
        Args:
            filepath: Path to audio file
            blocking: Wait for playback to finish
        """
        import scipy.io.wavfile as wav
        
        if filepath.endswith(".wav"):
            rate, data = wav.read(filepath)
            # Normalize if int16
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            self.play(data, rate, blocking)
        
        elif filepath.endswith(".mp3"):
            # Use pydub or ffmpeg for MP3
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(filepath)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                samples = samples / 32768.0  # Normalize
                self.play(samples, audio.frame_rate, blocking)
            except ImportError:
                raise ImportError("pydub required for MP3 playback. Run: pip install pydub")
        
        else:
            raise ValueError(f"Unsupported audio format: {filepath}")
    
    def play_bytes(self, audio_bytes: bytes, format: str = "mp3", blocking: bool = True):
        """
        Play audio from bytes.
        
        Args:
            audio_bytes: Raw audio bytes
            format: Audio format (mp3, wav)
            blocking: Wait for playback to finish
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            self.play_file(temp_path, blocking)
        finally:
            os.unlink(temp_path)
    
    def stop(self):
        """Stop current playback."""
        sd.stop()


def list_audio_devices():
    """List available audio devices."""
    if sd is None:
        print("sounddevice not installed")
        return
    
    print("Available audio devices:")
    print(sd.query_devices())


def get_default_input_device() -> Optional[dict]:
    """Get information about the default input device."""
    if sd is None:
        return None
    try:
        return sd.query_devices(kind='input')
    except sd.PortAudioError:
        return None


def get_default_output_device() -> Optional[dict]:
    """Get information about the default output device."""
    if sd is None:
        return None
    try:
        return sd.query_devices(kind='output')
    except sd.PortAudioError:
        return None
