"""
Text-to-Speech providers.
Pluggable architecture - add new providers by implementing TTSProvider base class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import tempfile
import subprocess

from config import TTSConfig


class TTSProvider(ABC):
    """Base class for all TTS providers. Implement this to add new TTS services."""
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: The text to synthesize
            
        Returns:
            Audio data as bytes (format depends on provider config)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name for logging."""
        pass


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS provider - high quality, requires API key."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.api_key = config.elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY environment variable "
                "or pass elevenlabs_api_key in TTSConfig."
            )
    
    def synthesize(self, text: str) -> bytes:
        import requests
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.elevenlabs_voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
        
        data = {
            "text": text,
            "model_id": self.config.elevenlabs_model,
            "voice_settings": {
                "stability": self.config.elevenlabs_stability,
                "similarity_boost": self.config.elevenlabs_similarity_boost,
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        return response.content
    
    def get_name(self) -> str:
        return "ElevenLabs"


class OpenAITTS(TTSProvider):
    """OpenAI TTS provider."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key in TTSConfig."
            )
    
    def synthesize(self, text: str) -> bytes:
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.audio.speech.create(
            model=self.config.openai_model,
            voice=self.config.openai_voice,
            input=text,
        )
        
        return response.content
    
    def get_name(self) -> str:
        return "OpenAI TTS"


class PiperTTS(TTSProvider):
    """Piper TTS - local, fast, free. Requires piper-tts installed."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.model_path = config.piper_model_path or self._get_default_model_path()
    
    def _get_default_model_path(self) -> str:
        """Get default model path based on model name."""
        home = os.path.expanduser("~")
        return os.path.join(home, ".local", "share", "piper", self.config.piper_model)
    
    def synthesize(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            # Run piper command
            process = subprocess.run(
                [
                    "piper",
                    "--model", self.model_path,
                    "--output_file", output_path,
                ],
                input=text.encode(),
                capture_output=True,
                check=True,
            )
            
            with open(output_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def get_name(self) -> str:
        return "Piper"


class CoquiTTS(TTSProvider):
    """Coqui TTS - local, high quality, more resource intensive."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self._tts = None
    
    def _get_tts(self):
        if self._tts is None:
            from TTS.api import TTS
            self._tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        return self._tts
    
    def synthesize(self, text: str) -> bytes:
        tts = self._get_tts()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            tts.tts_to_file(text=text, file_path=output_path)
            
            with open(output_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def get_name(self) -> str:
        return "Coqui TTS"


class SystemTTS(TTSProvider):
    """System TTS - uses OS built-in TTS. No setup required but basic quality."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
    
    def synthesize(self, text: str) -> bytes:
        import platform
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.run(
                    ["say", "-o", output_path, "--data-format=LEF32@22050", text],
                    check=True,
                )
            elif system == "Linux":
                subprocess.run(
                    ["espeak", "-w", output_path, text],
                    check=True,
                )
            elif system == "Windows":
                # Use PowerShell for Windows TTS
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.SetOutputToWaveFile("{output_path}")
                $synth.Speak("{text}")
                '''
                subprocess.run(["powershell", "-Command", ps_script], check=True)
            else:
                raise RuntimeError(f"Unsupported platform: {system}")
            
            with open(output_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def get_name(self) -> str:
        return "System TTS"


# Provider registry - add new providers here
TTS_PROVIDERS = {
    "elevenlabs": ElevenLabsTTS,
    "openai": OpenAITTS,
    "piper": PiperTTS,
    "coqui": CoquiTTS,
    "system": SystemTTS,
}


def get_tts_provider(config: TTSConfig) -> TTSProvider:
    """
    Factory function to get the appropriate TTS provider.
    
    Args:
        config: TTS configuration
        
    Returns:
        Initialized TTS provider instance
    """
    provider_class = TTS_PROVIDERS.get(config.provider)
    if provider_class is None:
        available = ", ".join(TTS_PROVIDERS.keys())
        raise ValueError(
            f"Unknown TTS provider: {config.provider}. Available: {available}"
        )
    
    return provider_class(config)
