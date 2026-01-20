"""
Wake word detection for hands-free activation.
Supports multiple wake word engines.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import os
import numpy as np
import threading
import time

from config import WakeWordConfig


class WakeWordDetector(ABC):
    """Base class for wake word detectors."""
    
    @abstractmethod
    def start(self, callback: Callable[[], None]):
        """
        Start listening for wake word.
        
        Args:
            callback: Function to call when wake word is detected
        """
        pass
    
    @abstractmethod
    def stop(self):
        """Stop listening for wake word."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return detector name."""
        pass


class PorcupineDetector(WakeWordDetector):
    """
    Picovoice Porcupine wake word detector.
    High accuracy, low resource usage, requires API key.
    
    Get free API key at: https://picovoice.ai/
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.access_key = config.porcupine_access_key or os.getenv("PORCUPINE_ACCESS_KEY")
        self._porcupine = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def _initialize(self):
        """Initialize Porcupine engine."""
        try:
            import pvporcupine
        except ImportError:
            raise ImportError(
                "pvporcupine required. Run: pip install pvporcupine"
            )
        
        if not self.access_key:
            raise ValueError(
                "Porcupine access key required. "
                "Set PORCUPINE_ACCESS_KEY environment variable or "
                "get free key at https://picovoice.ai/"
            )
        
        keyword = self.config.porcupine_keyword.lower()
        
        # Built-in keywords
        builtin_keywords = [
            "alexa", "americano", "blueberry", "bumblebee", "computer",
            "grapefruit", "grasshopper", "hey google", "hey siri", "jarvis",
            "ok google", "picovoice", "porcupine", "terminator"
        ]
        
        if keyword in builtin_keywords:
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[keyword],
                sensitivities=[self.config.porcupine_sensitivity],
            )
        else:
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[keyword],
                sensitivities=[self.config.porcupine_sensitivity],
            )
    
    def _detection_loop(self, callback: Callable[[], None]):
        """Main detection loop."""
        import sounddevice as sd
        
        frame_length = self._porcupine.frame_length
        sample_rate = self._porcupine.sample_rate
        
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=frame_length,
        ) as stream:
            while self._running:
                audio_data, _ = stream.read(frame_length)
                audio_frame = (audio_data.flatten() * 32767).astype(np.int16)
                keyword_index = self._porcupine.process(audio_frame)
                
                if keyword_index >= 0:
                    callback()
    
    def start(self, callback: Callable[[], None]):
        """Start listening for wake word."""
        self._initialize()
        self._running = True
        
        self._thread = threading.Thread(
            target=self._detection_loop,
            args=(callback,),
            daemon=True,
        )
        self._thread.start()
    
    def stop(self):
        """Stop listening."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None
    
    def get_name(self) -> str:
        return "Porcupine"


class OpenWakeWordDetector(WakeWordDetector):
    """
    OpenWakeWord detector - open source, runs locally, no API key needed.
    
    Install: pip install openwakeword
    """
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self._model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def _initialize(self):
        """Initialize OpenWakeWord."""
        try:
            from openwakeword import Model
        except ImportError:
            raise ImportError(
                "openwakeword required. Run: pip install openwakeword"
            )
        
        self._model = Model(
            wakeword_models=[self.config.openwakeword_model],
            inference_framework="onnx",
        )
    
    def _detection_loop(self, callback: Callable[[], None]):
        """Main detection loop."""
        import sounddevice as sd
        
        chunk_size = 1280
        
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype='int16',
            blocksize=chunk_size,
        ) as stream:
            while self._running:
                audio_data, _ = stream.read(chunk_size)
                audio_frame = audio_data.flatten()
                prediction = self._model.predict(audio_frame)
                
                for wakeword, scores in prediction.items():
                    if scores[-1] > self.config.openwakeword_threshold:
                        callback()
                        self._model.reset()
                        break
    
    def start(self, callback: Callable[[], None]):
        """Start listening for wake word."""
        self._initialize()
        self._running = True
        
        self._thread = threading.Thread(
            target=self._detection_loop,
            args=(callback,),
            daemon=True,
        )
        self._thread.start()
    
    def stop(self):
        """Stop listening."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def get_name(self) -> str:
        return "OpenWakeWord"


class KeyboardWakeDetector(WakeWordDetector):
    """
    Simple keyboard-based wake for testing.
    Press Enter to trigger.
    """
    
    def __init__(self, config: Optional[WakeWordConfig] = None):
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def _detection_loop(self, callback: Callable[[], None]):
        """Wait for Enter key."""
        while self._running:
            try:
                input()
                if self._running:
                    callback()
            except EOFError:
                break
    
    def start(self, callback: Callable[[], None]):
        """Start listening for Enter key."""
        self._running = True
        print("Press Enter to activate assistant...")
        
        self._thread = threading.Thread(
            target=self._detection_loop,
            args=(callback,),
            daemon=True,
        )
        self._thread.start()
    
    def stop(self):
        """Stop listening."""
        self._running = False
    
    def get_name(self) -> str:
        return "Keyboard"


WAKE_WORD_DETECTORS = {
    "porcupine": PorcupineDetector,
    "openwakeword": OpenWakeWordDetector,
    "keyboard": KeyboardWakeDetector,
}


def get_wake_word_detector(config: WakeWordConfig) -> WakeWordDetector:
    """Factory function to get wake word detector."""
    detector_class = WAKE_WORD_DETECTORS.get(config.provider)
    if detector_class is None:
        available = ", ".join(WAKE_WORD_DETECTORS.keys())
        raise ValueError(
            f"Unknown wake word provider: {config.provider}. Available: {available}"
        )
    
    return detector_class(config)
