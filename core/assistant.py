"""
Core Assistant - The main brain of the voice assistant.
Orchestrates STT, LLM, TTS, wake word detection, and workflows.
"""

import asyncio
import logging
import os
from typing import Optional, Callable
from enum import Enum
import time

from config import AssistantConfig, DEFAULT_CONFIG
from tts import get_tts_provider, TTSProvider
from stt import get_stt_provider, STTProvider
from llm import get_llm_provider, LLMProvider, IntentRouter
from workflows import (
    WorkflowManager,
    create_default_workflow_manager,
    WorkflowStatus,
    ConversationalWorkflow,
)
from .intent_cache import IntentCache
from .conversation import (
    ConversationContext,
    SessionManager,
    SqliteSessionStore,
    InMemorySessionStore,
    BackgroundTaskRunner,
)
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

        # Conversational layers (multi-turn agent framework)
        conv_cfg = self.config.conversation
        self._context_enabled = conv_cfg.context_enabled
        context_persist = (
            os.path.expanduser(conv_cfg.context_persist_path)
            if conv_cfg.context_enabled and not self.config.llm.ephemeral
            else None
        )
        self.context = ConversationContext(
            persist_path=context_persist,
            max_turns=conv_cfg.context_max_turns,
        )

        if conv_cfg.sessions_enabled:
            session_store = (
                InMemorySessionStore()
                if self.config.llm.ephemeral
                else SqliteSessionStore(os.path.expanduser(conv_cfg.session_store_path))
            )
            self.sessions: Optional[SessionManager] = SessionManager(
                store=session_store,
                workflows=self.workflows,
                default_timeout_s=conv_cfg.default_session_timeout_s,
                context=self.context,
            )
            # Workflows that manage their own WAITING sessions across new turns
            # (e.g. the reservation watcher's "stop watching" / "any luck")
            # need the shared store — WAITING sessions never take a user turn
            # directly, so control verbs reach them through the store.
            for wf in self.workflows.workflows.values():
                if hasattr(wf, "session_store") and wf.session_store is None:
                    wf.session_store = session_store
        else:
            self.sessions = None

        # LangGraph agent engine (opt-in via FRIDAY_AGENT_ENGINE=langgraph). Built
        # after the LLM, workflows, sessions, context and cache so it can reuse
        # them; None means Step 3 of process_input uses the legacy router.
        self.agent_engine = self._build_agent_engine()

        # Drives WAITING sessions + expiry sweep (+ agent wake-ups); started in
        # run(), stopped in stop(). Not started for ephemeral one-shot
        # interactions (--test / --chat).
        self.background_runner: Optional[BackgroundTaskRunner] = (
            BackgroundTaskRunner(
                self.sessions, self.speak, tick_seconds=conv_cfg.background_tick_seconds,
                agent_engine=self.agent_engine,
            )
            if self.sessions is not None and not self.config.llm.ephemeral
            else None
        )

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

        # Research substrate (see research/): set externally from main.py when
        # FRIDAY_RESEARCH=1. last_route records which pipeline step answered and
        # last_outcome whether the workflow it ran succeeded.
        self.research_recorder = None
        self.last_route: Optional[str] = None
        self.last_outcome: Optional[str] = None  # "success" | "failure" | None
    
    def _set_state(self, state: AssistantState):
        """Update state and notify callback."""
        self.state = state
        if self.on_state_change:
            self.on_state_change(state)
        
        if self.config.debug_mode:
            print(f"[State] {state.value}")
    
    def _log(self, message: str):
        """Log message if debug mode is enabled. Raw utterances can carry PII
        (names, numbers, even card digits read aloud) — always redact (§8 L1)."""
        if self.config.debug_mode:
            from core.harness import redact_text
            print(f"[Debug] {redact_text(message)}")

    def _build_agent_engine(self):
        """Construct the LangGraph AgentEngine when the config asks for it and
        the libraries are installed; otherwise None (legacy router). Any failure
        here is a warning, never a crash — Friday always boots."""
        agent_cfg = getattr(self.config, "agent", None)
        if agent_cfg is None or agent_cfg.engine != "langgraph":
            return None
        try:
            import agent as agent_pkg
            if not agent_pkg.is_available():
                print(f"⚠️  FRIDAY_AGENT_ENGINE=langgraph but the engine is unavailable "
                      f"({agent_pkg.unavailable_reason()}); using the legacy router.")
                return None
            engine = agent_pkg.AgentEngine.from_assistant(self)
            self._log(f"Agent engine: {engine.describe()}")
            return engine
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Agent engine failed to start; using the legacy router: %s", e, exc_info=True)
            print(f"⚠️  Agent engine failed to start ({e}); using the legacy router.")
            return None

    @property
    def engine_label(self) -> str:
        return "langgraph" if self.agent_engine is not None else "legacy router"
    
    async def _maybe_start_session(self, workflow, text: str, entities: dict, user_id: str):
        """If `workflow` is conversational, open a multi-turn session and return its
        first spoken message. Returns None when the workflow is single-shot."""
        if self.sessions is not None and isinstance(workflow, ConversationalWorkflow):
            turn = await self.sessions.open(workflow, text, entities, user_id)
            self.context.update(workflow.name, entities, text)
            return turn.message
        return None

    def _handle_workflow_failure(self, text: str, workflow_name: str, result) -> str:
        failure_context = (
            f"The user asked: '{text}'. "
            f"The {workflow_name} system responded with an error: "
            f"{result.error or result.message}"
        )
        # record_research=False: the prompt above is synthetic, so recording it
        # would put a sentence the user never said into the study's corpus and
        # from there into the eval split and the SFT dataset.
        self.last_outcome = "failure"
        return self.llm.generate_response(failure_context, record_research=False)

    async def process_input(self, text: str, user_id: str = "default",
                            channel: Optional[str] = None) -> str:
        """Process user input and generate a response (see _process_input_inner).

        Thin wrapper that times the turn and, when the research substrate is
        enabled, records every exchange with the route that answered it. The
        recorder call must never affect the reply.
        """
        self.last_route = None
        self.last_outcome = None
        start = time.monotonic()
        reply = await self._process_input_inner(text, user_id)
        if self.research_recorder is not None:
            try:
                self.research_recorder.record_turn(
                    text,
                    reply,
                    route=self.last_route or "unknown",
                    latency_ms=int((time.monotonic() - start) * 1000),
                    user_id=user_id,
                    channel=channel,
                    outcome=self.last_outcome,
                )
            except Exception:
                self._log("Research recorder failed for turn.")
        return reply

    async def _process_input_inner(self, text: str, user_id: str = "default") -> str:
        """
        Process user input and generate a response.

        Pipeline:
        0. Active multi-turn session   → route the turn into it (global "cancel" aborts)
           Pending agent confirmation → this turn answers it (global "cancel" abandons)
        1. Fast keyword/pattern match  → conversational? open session, else execute
        2. Semantic intent cache       → conversational? open session, else execute
        3. Agent engine (LangGraph)    → tool-calling loop over the workflow registry
           └ on any error, or when the engine is off: legacy Claude routing prompt
              → classify (context-enriched), open/execute workflow
              └ no workflow found      → Claude generates a conversational response
        """
        self._log(f"Processing: {text}")

        # Step 0: an active dialogue session takes the turn.
        if self.sessions is not None and self.sessions.has_active(user_id):
            self.last_route = "session"
            if self.sessions.is_global_escape(text):
                self.sessions.cancel(user_id, "user aborted")
                return "Very well, sir. I've set that aside."
            self._log("Routing to active session")
            result = await self.sessions.handle(user_id, text)
            return result.message

        # Step 0b: a pending agent confirmation ("Shall I unlock the back door?")
        # owns the turn, exactly like an active session — otherwise "yes please"
        # could keyword-match a workflow instead of answering the question. At
        # most one of (legacy session, graph interrupt) is pending per user:
        # the start_task handoff ends the graph turn without an interrupt.
        if self.agent_engine is not None:
            try:
                if await self.agent_engine.has_pending_interrupt(user_id):
                    if SessionManager.is_global_escape(text):
                        await self.agent_engine.cancel(user_id)
                        return "Very well, sir. I've set that aside."
                    self._log("Routing to pending agent confirmation")
                    return await self.agent_engine.handle(text, user_id, cacheable=False)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Agent engine failed on a pending confirmation; continuing with the "
                    "legacy pipeline: %s", e, exc_info=True)
                self._log(f"Agent engine error (pending turn): {e}")

        # Layer A: enrich for routing continuity (raw text is kept for keyword/cache
        # matching so an injected context prefix can't cause false keyword hits).
        enriched = self.context.enrich(text) if self._context_enabled else text
        if enriched != text:
            self._log(f"Context-enriched: {enriched}")

        # Step 1: fast keyword/pattern matching
        matching_workflow = self.workflows.find_matching_workflow(text)
        if matching_workflow:
            self._log(f"Keyword match: {matching_workflow.name}")
            self.last_route = f"keyword:{matching_workflow.name}"
            started = await self._maybe_start_session(matching_workflow, text, {}, user_id)
            if started is not None:
                return started
            result = await matching_workflow.execute(text, {})
            if result.status == WorkflowStatus.SUCCESS:
                self.last_outcome = "success"
                self.context.update(matching_workflow.name, {}, text)
                return result.message
            elif result.status == WorkflowStatus.FAILURE:
                return self._handle_workflow_failure(text, matching_workflow.name, result)
            return result.message

        # Step 2: semantic intent cache
        if self.intent_cache:
            cached = self.intent_cache.query(text)
            if cached:
                workflow_name, entities = cached
                workflow = self.workflows.workflows.get(workflow_name)
                if workflow:
                    self._log(f"Cache hit: {workflow_name}")
                    self.last_route = f"cache:{workflow_name}"
                    started = await self._maybe_start_session(workflow, text, entities, user_id)
                    if started is not None:
                        return started
                    result = await workflow.execute(text, entities)
                    if result.status == WorkflowStatus.SUCCESS:
                        self.last_outcome = "success"
                        self.context.update(workflow_name, entities, text)
                        return result.message
                    elif result.status == WorkflowStatus.FAILURE:
                        return self._handle_workflow_failure(text, workflow_name, result)
                    return result.message

        # Step 3: agent engine (LangGraph). Any exception falls through to the
        # legacy router below, so Friday keeps working if the engine misbehaves.
        if self.agent_engine is not None:
            try:
                self._log("Routing via agent engine")
                return await self.agent_engine.handle(text, user_id, cacheable=(enriched == text))
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Agent engine failed; falling back to the legacy router: %s", e, exc_info=True)
                self._log(f"Agent engine error: {e}")

        # Step 3 (legacy): Claude routing (enhanced fallback, uses context-enriched text)
        self._log("Routing via Claude")
        route = self.intent_router.route(enriched, self.workflows)

        if route.workflow_name:
            workflow = self.workflows.workflows.get(route.workflow_name)
            if workflow:
                self.last_route = f"router:{route.workflow_name}"
                started = await self._maybe_start_session(workflow, text, route.entities, user_id)
                if started is not None:
                    return started
                result = await workflow.execute(text, route.entities)
                if result.status == WorkflowStatus.SUCCESS:
                    self.last_outcome = "success"
                    # Only cache routes decided from the raw utterance. When the
                    # router saw context-enriched text (enriched != text), the
                    # decision may reflect transient prior-turn context — caching
                    # it against the raw text permanently poisons the cache (e.g.
                    # "hey" → time after a time query). Follow-ups are context-
                    # dependent by nature and shouldn't be cached anyway.
                    if self.intent_cache and enriched == text:
                        self.intent_cache.store(text, route.workflow_name, route.entities)
                    self.context.update(route.workflow_name, route.entities, text)
                    return result.message
                elif result.status == WorkflowStatus.FAILURE:
                    return self._handle_workflow_failure(text, route.workflow_name, result)
                return result.message

        # No workflow matched — answer conversationally.
        #
        # This must go through the LLM provider, never a reply drafted by the
        # router. The router runs a bare classifier prompt: no personality, no
        # conversation history, no remembered facts, no search. Short-circuiting
        # on its draft made every free-chat turn sound like a help menu, and it
        # also bypassed the provider's exchange recording, which is the only
        # place the research substrate captures the context snapshot.
        self.last_route = "chat"
        return self.llm.generate_response(text)
    
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
            # Warning (not debug): a broken TTS otherwise fails in total
            # silence — every channel just goes mute with nothing in the logs.
            logging.getLogger(__name__).warning("TTS failed, reply not spoken: %s", e)
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
    
    def start_listening(self, allow_keyboard_fallback: bool = True) -> bool:
        """
        Start the session runner and wake-word detector without blocking.

        Returns True if a voice trigger (wake word or, when allowed, keyboard)
        is active. Callers that want to own stdin themselves (e.g. a concurrent
        text-chat loop) should pass allow_keyboard_fallback=False so the mic-less
        case quietly disables voice instead of grabbing stdin.
        """
        self._running = True

        # Start the multi-turn background runner (WAITING sessions + expiry sweep)
        if self.background_runner is not None:
            self.background_runner.start()

        # Initialize wake word detector
        try:
            self._wake_detector = get_wake_word_detector(self.config.wake_word)
            self._wake_detector.start(self._on_wake_word_detected)
            print(f"Listening for wake word '{self.config.wake_word.porcupine_keyword}'...")
            return True
        except Exception as e:
            if allow_keyboard_fallback:
                print(f"Wake word detection failed: {e}")
                print("Falling back to keyboard activation (press Enter)...")
                from utils import KeyboardWakeDetector
                self._wake_detector = KeyboardWakeDetector()
                self._wake_detector.start(self._on_wake_word_detected)
                return True
            print(f"⚠️  Voice disabled (wake word unavailable): {e}")
            self._wake_detector = None
            return False

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

        self.start_listening(allow_keyboard_fallback=True)
        print("(Press Ctrl+C to quit)\n")

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

        if self.background_runner is not None:
            self.background_runner.stop()

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
        """Clear the conversation history (legacy list and, when on, the agent thread)."""
        self.llm.clear_history()
        if self.agent_engine is not None:
            self.agent_engine.reset()


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
