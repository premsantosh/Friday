"""
Speech-to-Text providers.
Pluggable architecture - add new providers by implementing STTProvider base class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import tempfile
import numpy as np

from config import STTConfig


class STTProvider(ABC):
    """Base class for all STT providers. Implement this to add new STT services."""
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Convert audio to text.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name for logging."""
        pass


class WhisperLocalSTT(STTProvider):
    """OpenAI Whisper running locally - free, private, good quality."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            import whisper
            
            device = self.config.whisper_device
            if device == "auto":
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self._model = whisper.load_model(
                self.config.whisper_model,
                device=device,
            )
        return self._model
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        model = self._get_model()
        
        # Whisper expects float32 audio normalized to [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0  # Normalize from int16
        
        # Resample to 16kHz if needed (Whisper requirement)
        if sample_rate != 16000:
            import scipy.signal
            audio_data = scipy.signal.resample(
                audio_data,
                int(len(audio_data) * 16000 / sample_rate)
            )
        
        result = model.transcribe(
            audio_data,
            language=self.config.whisper_language,
            fp16=False,  # More compatible
        )
        
        return result["text"].strip()
    
    def get_name(self) -> str:
        return f"Whisper Local ({self.config.whisper_model})"


class WhisperAPISTT(STTProvider):
    """OpenAI Whisper API - cloud-based, fast, costs money."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required for Whisper API. "
                "Set OPENAI_API_KEY environment variable."
            )
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        from openai import OpenAI
        import scipy.io.wavfile
        
        client = OpenAI(api_key=self.api_key)
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Ensure proper format
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            scipy.io.wavfile.write(temp_path, sample_rate, audio_data)
            
            with open(temp_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.config.whisper_language,
                )
            
            return result.text.strip()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_name(self) -> str:
        return "Whisper API"


class VoskSTT(STTProvider):
    """Vosk STT - local, lightweight, fast, free."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            from vosk import Model
            
            # Vosk models need to be downloaded separately
            model_path = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Vosk model not found at {model_path}. "
                    "Download from https://alphacephei.com/vosk/models"
                )
            
            self._model = Model(model_path)
        return self._model
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        from vosk import KaldiRecognizer
        import json
        
        model = self._get_model()
        recognizer = KaldiRecognizer(model, sample_rate)
        
        # Convert to bytes
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        audio_bytes = audio_data.tobytes()
        
        recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(recognizer.FinalResult())
        
        return result.get("text", "").strip()
    
    def get_name(self) -> str:
        return "Vosk"


class DeepgramSTT(STTProvider):
    """Deepgram STT - cloud-based, very fast, real-time capable."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.api_key = config.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key required. Set DEEPGRAM_API_KEY environment variable."
            )
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        import requests
        import scipy.io.wavfile
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            scipy.io.wavfile.write(temp_path, sample_rate, audio_data)
            
            with open(temp_path, "rb") as audio_file:
                response = requests.post(
                    "https://api.deepgram.com/v1/listen",
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "audio/wav",
                    },
                    data=audio_file.read(),
                    params={
                        "model": "nova-2",
                        "language": "en",
                    },
                )
            
            response.raise_for_status()
            result = response.json()
            
            return result["results"]["channels"][0]["alternatives"][0]["transcript"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_name(self) -> str:
        return "Deepgram"


# Provider registry
STT_PROVIDERS = {
    "whisper": WhisperLocalSTT,
    "whisper_api": WhisperAPISTT,
    "vosk": VoskSTT,
    "deepgram": DeepgramSTT,
}


def get_stt_provider(config: STTConfig) -> STTProvider:
    """
    Factory function to get the appropriate STT provider.
    
    Args:
        config: STT configuration
        
    Returns:
        Initialized STT provider instance
    """
    provider_class = STT_PROVIDERS.get(config.provider)
    if provider_class is None:
        available = ", ".join(STT_PROVIDERS.keys())
        raise ValueError(
            f"Unknown STT provider: {config.provider}. Available: {available}"
        )
    
    return provider_class(config)
