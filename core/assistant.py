"""
Core Assistant - The main brain of the voice assistant.
Orchestrates STT, LLM, TTS, wake word detection, and workflows.
"""

import asyncio
from typing import Optional, Callable
from enum import Enum
import time

from config import AssistantConfig, DEFAULT_CONFIG
from tts import get_tts_provider, TTSProvider
from stt import get_stt_provider, STTProvider
from llm import get_llm_provider, LLMProvider
from workflows import WorkflowManager, create_default_workflow_manager, WorkflowStatus
from utils import (
    AudioRecorder,
    AudioPlayer,
    AudioConfig,
    get_wake_word_detector,
    WakeWordDetector,
)


class AssistantState(Enum):
    """Current state of the assistant."""
    IDLE = "idle"           # Waiting for wake word
    LISTENING = "listening"  # Recording user speech
    THINKING = "thinking"    # Processing with LLM
    SPEAKING = "speaking"    # Playing TTS response
    ERROR = "error"          # Something went wrong


class VoiceAssistant:
    """
    Main voice assistant class.
    
    Usage:
        assistant = VoiceAssistant()
        assistant.run()  # Blocking
        
    Or for async:
        assistant = VoiceAssistant()
        await assistant.run_async()
    """
    
    def __init__(
        self,
        config: Optional[AssistantConfig] = None,
        workflow_manager: Optional[WorkflowManager] = None,
    ):
        """
        Initialize the voice assistant.
        
        Args:
            config: Configuration settings (uses DEFAULT_CONFIG if not provided)
            workflow_manager: Custom workflow manager (creates default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.tts: TTSProvider = get_tts_provider(self.config.tts)
        self.stt: STTProvider = get_stt_provider(self.config.stt)
        self.llm: LLMProvider = get_llm_provider(self.config.llm, self.config.personality)
        
        # Workflow manager for smart home and other capabilities
        self.workflows = workflow_manager or create_default_workflow_manager()
        
        # Audio components
        self.recorder = AudioRecorder(AudioConfig(
            sample_rate=self.config.stt.input_sample_rate,
            channels=self.config.stt.input_channels,
            silence_threshold=self.config.stt.silence_threshold,
            silence_duration=self.config.stt.silence_duration,
        ))
        self.player = AudioPlayer()
        
        # Wake word detector (initialized lazily)
        self._wake_detector: Optional[WakeWordDetector] = None
        
        # State management
        self.state = AssistantState.IDLE
        self._running = False
        
        # Callbacks for UI integration
        self.on_state_change: Optional[Callable[[AssistantState], None]] = None
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def _set_state(self, state: AssistantState):
        """Update state and notify callback."""
        self.state = state
        if self.on_state_change:
            self.on_state_change(state)
        
        if self.config.debug_mode:
            print(f"[State] {state.value}")
    
    def _log(self, message: str):
        """Log message if debug mode is enabled."""
        if self.config.debug_mode:
            print(f"[Debug] {message}")
    
    async def process_input(self, text: str) -> str:
        """
        Process user input and generate response.
        
        This is the core processing pipeline:
        1. Check for matching workflow
        2. If workflow matches, execute it
        3. Otherwise, send to LLM
        4. Return response text
        
        Args:
            text: User's transcribed speech
            
        Returns:
            Response text to speak
        """
        self._log(f"Processing: {text}")
        
        # Check if any workflow matches
        matching_workflow = self.workflows.find_matching_workflow(text)
        
        if matching_workflow:
            self._log(f"Matched workflow: {matching_workflow.name}")
            
            # Extract entities (simplified - in production you'd use NLU)
            entities = self._extract_entities(text)
            
            # Execute workflow
            result = await matching_workflow.execute(text, entities)
            
            if result.status == WorkflowStatus.SUCCESS:
                return result.message
            elif result.status == WorkflowStatus.FAILURE:
                # Let LLM handle the failure gracefully
                failure_context = f"The user asked: '{text}'. The {matching_workflow.name} system responded with an error: {result.error or result.message}"
                return self.llm.generate_response(failure_context)
            else:
                return result.message
        
        # No workflow matched - send to LLM
        # Include workflow context so LLM knows what it can do
        workflow_context = self.workflows.get_all_context_for_llm()
        
        # Add context about available capabilities
        enhanced_input = text
        if workflow_context and "No special capabilities" not in workflow_context:
            # The LLM already has the personality prompt
            # Just send the user input
            pass
        
        response = self.llm.generate_response(text)
        return response
    
    def _extract_entities(self, text: str) -> dict:
        """
        Simple entity extraction from text.
        In production, you'd use proper NLU or let the LLM extract entities.
        """
        import re
        
        entities = {}
        text_lower = text.lower()
        
        # Extract action
        if any(word in text_lower for word in ["turn on", "switch on", "enable"]):
            entities["action"] = "on"
        elif any(word in text_lower for word in ["turn off", "switch off", "disable"]):
            entities["action"] = "off"
        elif "dim" in text_lower:
            entities["action"] = "dim"
        elif "lock" in text_lower and "unlock" not in text_lower:
            entities["action"] = "lock"
        elif "unlock" in text_lower:
            entities["action"] = "unlock"
        elif "check" in text_lower or "who" in text_lower:
            entities["action"] = "check"
        
        # Extract room/location
        rooms = ["living room", "bedroom", "kitchen", "bathroom", "office", "garage", "basement", "attic"]
        for room in rooms:
            if room in text_lower:
                entities["room"] = room
                break
        
        # Extract door
        doors = ["front", "back", "side", "garage"]
        for door in doors:
            if door in text_lower:
                entities["door"] = door
                break
        
        # Extract mood
        mood_keywords = {
            "romantic": ["romantic", "romance", "date night", "intimate", "candlelight"],
            "relax": ["relax", "relaxing", "chill", "calm", "unwind", "wind down", "peaceful"],
            "energize": ["energize", "energetic", "energy", "pump up", "motivated", "productive"],
            "party": ["party", "dance", "celebrate", "celebration", "fiesta"],
            "bedtime": ["bed", "sleep", "bedtime", "good night", "goodnight", "going to bed", "sleepy", "night night"],
            "focus": ["focus", "concentrate", "study", "reading", "work mode"],
            "movie": ["movie", "cinema", "film", "movie night", "watching"],
            "morning": ["morning", "wake up", "sunrise", "good morning"],
        }
        for mood, keywords in mood_keywords.items():
            if any(kw in text_lower for kw in keywords):
                entities["mood"] = mood
                entities["action"] = "mood"
                break

        # Extract numbers (brightness, temperature)
        numbers = re.findall(r'\d+', text)
        if numbers:
            num = int(numbers[0])
            if num <= 100:
                entities["brightness"] = num
            else:
                entities["temperature"] = num

        return entities
    
    def speak(self, text: str):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
        """
        self._set_state(AssistantState.SPEAKING)
        
        try:
            # Generate audio
            audio_bytes = self.tts.synthesize(text)
            
            # Play audio
            self.player.play_bytes(audio_bytes, format=self.tts.audio_format)
            
        except Exception as e:
            self._log(f"TTS error: {e}")
            if self.on_error:
                self.on_error(f"Speech synthesis failed: {e}")
        
        finally:
            self._set_state(AssistantState.IDLE)
    
    def listen(self) -> Optional[str]:
        """
        Listen for user speech and transcribe it.
        
        Returns:
            Transcribed text, or None if nothing detected
        """
        self._set_state(AssistantState.LISTENING)
        
        try:
            # Record audio
            audio_data = self.recorder.record_until_silence(
                on_speech_start=lambda: self._log("Speech detected..."),
                on_speech_end=lambda: self._log("Speech ended."),
            )
            
            if len(audio_data) < 1000:  # Too short
                self._log("Audio too short, ignoring.")
                return None
            
            self._set_state(AssistantState.THINKING)
            
            # Transcribe
            text = self.stt.transcribe(audio_data, self.config.stt.input_sample_rate)
            
            if text:
                self._log(f"Transcribed: {text}")
                if self.on_transcript:
                    self.on_transcript(text)
            
            return text if text.strip() else None
            
        except Exception as e:
            self._log(f"STT error: {e}")
            if self.on_error:
                self.on_error(f"Speech recognition failed: {e}")
            return None
    
    async def handle_activation(self):
        """
        Handle a single activation cycle:
        1. Listen for speech
        2. Process input
        3. Speak response
        """
        # Listen
        text = self.listen()
        
        if not text:
            # Nothing detected, give feedback
            self.speak("I didn't catch that, sir.")
            return
        
        # Process
        self._set_state(AssistantState.THINKING)
        
        try:
            response = await self.process_input(text)
            
            if self.on_response:
                self.on_response(response)
            
            # Speak response
            self.speak(response)
            
        except Exception as e:
            self._log(f"Processing error: {e}")
            self.speak("I apologize, sir, but I encountered an error processing that request.")
            if self.on_error:
                self.on_error(str(e))
    
    def _on_wake_word_detected(self):
        """Callback when wake word is detected."""
        self._log("Wake word detected!")
        
        # Run activation in async context
        asyncio.run(self.handle_activation())
    
    def run(self):
        """
        Run the assistant (blocking).
        Listens for wake word and processes commands.
        """
        print(f"\n{'='*50}")
        print(f"  {self.config.personality.name} Voice Assistant")
        print(f"{'='*50}")
        print(f"  TTS: {self.tts.get_name()}")
        print(f"  STT: {self.stt.get_name()}")
        print(f"  LLM: {self.llm.get_name()}")
        print(f"  Wake word: {self.config.wake_word.porcupine_keyword}")
        print(f"{'='*50}\n")
        
        self._running = True
        
        # Initialize wake word detector
        try:
            self._wake_detector = get_wake_word_detector(self.config.wake_word)
            self._wake_detector.start(self._on_wake_word_detected)
            print(f"Listening for wake word '{self.config.wake_word.porcupine_keyword}'...")
            print("(Press Ctrl+C to quit)\n")
        except Exception as e:
            print(f"Wake word detection failed: {e}")
            print("Falling back to keyboard activation (press Enter)...")
            from utils import KeyboardWakeDetector
            self._wake_detector = KeyboardWakeDetector()
            self._wake_detector.start(self._on_wake_word_detected)
        
        # Keep running until interrupted
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def run_single_interaction(self, text: str) -> str:
        """
        Process a single text interaction (no voice).
        Useful for testing or text-based interfaces.
        
        Args:
            text: User input text
            
        Returns:
            Assistant response text
        """
        return asyncio.run(self.process_input(text))
    
    def stop(self):
        """Stop the assistant."""
        self._running = False
        
        if self._wake_detector:
            self._wake_detector.stop()
            self._wake_detector = None
        
        self._set_state(AssistantState.IDLE)
        print("Assistant stopped.")
    
    def update_personality(self, **kwargs):
        """
        Update personality settings on the fly.

        Example:
            assistant.update_personality(sarcasm_level=SarcasmLevel.HEAVY)
        """
        from config import PersonalityConfig

        for key, value in kwargs.items():
            if hasattr(self.config.personality, key):
                setattr(self.config.personality, key, value)

        # Regenerate LLM prompt
        self.llm.update_personality(self.config.personality)

        self._log(f"Personality updated: {kwargs}")

    def clear_history(self):
        """Clear the conversation history."""
        self.llm.clear_history()


def create_assistant(
    name: str = "Jarvis",
    sarcasm: str = "moderate",
    tts_provider: str = "elevenlabs",
    llm_provider: str = "anthropic",
    **kwargs,
) -> VoiceAssistant:
    """
    Convenience function to create an assistant with common settings.
    
    Args:
        name: Assistant name
        sarcasm: Sarcasm level (none, light, moderate, heavy, maximum)
        tts_provider: TTS provider name
        llm_provider: LLM provider name
        **kwargs: Additional config overrides
        
    Returns:
        Configured VoiceAssistant instance
    """
    from config import (
        AssistantConfig,
        PersonalityConfig,
        TTSConfig,
        STTConfig,
        LLMConfig,
        WakeWordConfig,
        SarcasmLevel,
        FormalityLevel,
        WarmthLevel,
    )
    
    # Map sarcasm string to enum
    sarcasm_map = {
        "none": SarcasmLevel.NONE,
        "light": SarcasmLevel.LIGHT,
        "moderate": SarcasmLevel.MODERATE,
        "heavy": SarcasmLevel.HEAVY,
        "maximum": SarcasmLevel.MAXIMUM,
    }
    
    config = AssistantConfig(
        personality=PersonalityConfig(
            name=name,
            sarcasm_level=sarcasm_map.get(sarcasm, SarcasmLevel.MODERATE),
            formality_level=FormalityLevel.BUTLER,
            warmth_level=WarmthLevel.WARM,
            wit_enabled=True,
            use_british_vocabulary=True,
        ),
        tts=TTSConfig(provider=tts_provider),
        stt=STTConfig(provider="whisper"),
        llm=LLMConfig(provider=llm_provider),
        wake_word=WakeWordConfig(
            provider="porcupine",
            porcupine_keyword=name.lower() if name.lower() in ["jarvis", "alexa", "computer"] else "jarvis",
        ),
        debug_mode=kwargs.get("debug", False),
    )
    
    return VoiceAssistant(config)
