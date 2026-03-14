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
from llm import get_llm_provider, LLMProvider, IntentRouter
from workflows import WorkflowManager, create_default_workflow_manager, WorkflowStatus
from .intent_cache import IntentCache
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

        # Semantic intent cache and router (enhanced fallback)
        cache_cfg = self.config.intent_cache
        self.intent_cache = IntentCache(
            path=cache_cfg.path,
            collection_name=cache_cfg.collection_name,
            similarity_threshold=cache_cfg.similarity_threshold,
        ) if cache_cfg.enabled else None
        self.intent_router = IntentRouter(self.config.llm)
        
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
        Process user input and generate a response.

        Pipeline:
        1. Fast keyword/pattern match  → execute workflow directly
        2. Semantic intent cache       → execute cached workflow routing
        3. Claude routing prompt       → classify intent, store result, execute workflow
           └ no workflow found         → Claude generates a conversational response
        """
        self._log(f"Processing: {text}")

        # Step 1: fast keyword/pattern matching
        matching_workflow = self.workflows.find_matching_workflow(text)
        if matching_workflow:
            self._log(f"Keyword match: {matching_workflow.name}")
            result = await matching_workflow.execute(text, {})
            if result.status == WorkflowStatus.SUCCESS:
                return result.message
            elif result.status == WorkflowStatus.FAILURE:
                failure_context = (
                    f"The user asked: '{text}'. "
                    f"The {matching_workflow.name} system responded with an error: "
                    f"{result.error or result.message}"
                )
                return self.llm.generate_response(failure_context)
            return result.message

        # Step 2: semantic intent cache
        if self.intent_cache:
            cached = self.intent_cache.query(text)
            if cached:
                workflow_name, entities = cached
                workflow = self.workflows.workflows.get(workflow_name)
                if workflow:
                    self._log(f"Cache hit: {workflow_name}")
                    result = await workflow.execute(text, entities)
                    if result.status == WorkflowStatus.SUCCESS:
                        return result.message
                    elif result.status == WorkflowStatus.FAILURE:
                        failure_context = (
                            f"The user asked: '{text}'. "
                            f"The {workflow_name} system responded with an error: "
                            f"{result.error or result.message}"
                        )
                        return self.llm.generate_response(failure_context)
                    return result.message

        # Step 3: Claude routing (enhanced fallback)
        self._log("Routing via Claude")
        route = self.intent_router.route(text, self.workflows)

        if route.workflow_name:
            workflow = self.workflows.workflows.get(route.workflow_name)
            if workflow:
                result = await workflow.execute(text, route.entities)
                if result.status == WorkflowStatus.SUCCESS:
                    if self.intent_cache:
                        self.intent_cache.store(text, route.workflow_name, route.entities)
                    return result.message
                elif result.status == WorkflowStatus.FAILURE:
                    failure_context = (
                        f"The user asked: '{text}'. "
                        f"The {route.workflow_name} system responded with an error: "
                        f"{result.error or result.message}"
                    )
                    return self.llm.generate_response(failure_context)
                return result.message

        # No workflow matched — use Claude's conversational response or fall back
        return route.response or self.llm.generate_response(text)
    
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
