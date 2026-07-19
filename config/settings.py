"""
Configuration settings for the voice assistant.
All personality and service settings are tunable here.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SarcasmLevel(Enum):
    """How sarcastic the assistant should be."""
    NONE = 0        # Completely professional
    LIGHT = 1       # Occasional gentle teasing
    MODERATE = 2    # Regular witty remarks
    HEAVY = 3       # Constant roasting (use with caution)
    MAXIMUM = 4     # Full GLaDOS mode


class FormalityLevel(Enum):
    """How formal the assistant's speech should be."""
    CASUAL = 0      # "Hey! What's up?"
    FRIENDLY = 1    # "Hi there! How can I help?"
    PROFESSIONAL = 2  # "Good morning. How may I assist?"
    FORMAL = 3      # "Good morning, sir. How may I be of service?"
    BUTLER = 4      # "Good morning, sir. I trust you slept adequately."


class WarmthLevel(Enum):
    """How warm and caring vs cold and efficient."""
    COLD = 0        # Pure efficiency, no emotion
    NEUTRAL = 1     # Polite but distant
    WARM = 2        # Friendly and caring
    AFFECTIONATE = 3  # Genuinely invested in user's wellbeing


@dataclass
class PersonalityConfig:
    """
    All tunable personality parameters for the assistant.
    Modify these to change how your assistant behaves.
    """
    # Basic identity
    name: str = "Jarvis"
    user_title: str = "sir"  # How the assistant addresses you: sir, ma'am, boss, etc.
    
    # Personality sliders
    sarcasm_level: SarcasmLevel = SarcasmLevel.MODERATE
    formality_level: FormalityLevel = FormalityLevel.BUTLER
    warmth_level: WarmthLevel = WarmthLevel.WARM
    
    # Wit and humor
    wit_enabled: bool = True
    self_aware_ai_jokes: bool = True  # Jokes about being an AI
    observational_humor: bool = True  # Comments on user's habits
    
    # Speech patterns
    use_british_vocabulary: bool = True
    use_contractions: bool = False  # "I'm" vs "I am"
    max_response_sentences: int = 3  # For simple requests
    
    # Behavior modifiers
    sass_timeout_on_stress: bool = True  # Reduce sarcasm if user seems upset
    urgent_mode_override: bool = True    # Be direct for safety/urgent matters
    
    # Topics
    off_limits_topics: list = field(default_factory=list)  # Topics to never joke about
    
    # Home appliance / device knowledge baked into the system prompt.
    # Populated at startup by integrations that want the LLM to have persistent
    # background knowledge (e.g. "coffee machine takes 20 min to warm up").
    home_context: str = ""

    # Custom phrases the assistant likes to use
    favorite_phrases: list = field(default_factory=lambda: [
        "Indeed",
        "Certainly",
        "I shall endeavour",
        "Might I suggest",
        "As you wish",
        "Very good",
        "I have taken the liberty",
        "If I may be so bold",
        "I trust",
        "One does find",
        "Most satisfactory",
        "I am at your disposal",
        "Quite so",
        "I dare say",
    ])


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    # Provider selection
    provider: str = "piper"  # Options: piper, elevenlabs, openai, coqui, system
    
    # ElevenLabs settings
    elevenlabs_api_key: Optional[str] = None  # Set via environment variable
    elevenlabs_voice_id: Optional[str] = None  # Set via ELEVENLABS_VOICE_ID env var
    elevenlabs_model: str = "eleven_monolingual_v1"
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    
    # OpenAI TTS settings
    openai_api_key: Optional[str] = None
    openai_voice: str = "onyx"  # Options: alloy, echo, fable, onyx, nova, shimmer
    openai_model: str = "tts-1"
    
    # Piper (local) settings
    piper_model: str = "en_GB-northern_english_male-medium.onnx"  # British English voice
    piper_model_path: Optional[str] = None
    
    # Audio settings
    output_sample_rate: int = 22050
    output_format: str = "mp3"


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    # Provider selection
    provider: str = "whisper"  # Options: whisper, whisper_api, vosk, deepgram
    
    # Whisper (local) settings
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_language: str = "en"
    whisper_device: str = "auto"  # auto, cpu, cuda, mps
    
    # Whisper API settings
    openai_api_key: Optional[str] = None
    
    # Deepgram settings
    deepgram_api_key: Optional[str] = None
    
    # Audio input settings
    input_sample_rate: int = 16000
    input_channels: int = 1
    silence_threshold: float = 0.01
    silence_duration: float = 1.0  # Seconds of silence to stop recording


@dataclass
class LLMConfig:
    """Language Model configuration."""
    # Provider selection
    provider: str = "anthropic"  # Options: anthropic, openai, ollama
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    
    # Ollama (local) settings
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    extractor_model: str = "phi3:mini"  # Local model used for background fact extraction
    
    # Generation settings
    max_tokens: int = 150
    temperature: float = 0.7
    max_history: int = 10  # Max conversation turns to keep (0 = unlimited)

    # When True, no data is read from or written to any persistent store.
    # Used by --chat and --test modes so test sessions don't pollute long-term memory.
    ephemeral: bool = False


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""
    # Provider selection
    provider: str = "porcupine"  # Options: porcupine, openwakeword
    
    # Porcupine settings
    porcupine_access_key: Optional[str] = None
    porcupine_keyword: str = "jarvis"  # Built-in: jarvis, alexa, computer, etc.
    porcupine_sensitivity: float = 0.5
    
    # OpenWakeWord settings
    openwakeword_model: str = "hey_jarvis"
    openwakeword_threshold: float = 0.5


@dataclass
class SearchConfig:
    """Web search configuration."""
    enabled: bool = True
    classifier_model: str = "phi3:mini"
    classifier_timeout: int = 10
    provider: str = "tavily"
    tavily_api_key: Optional[str] = None
    max_results: int = 5


@dataclass
class IntentCacheConfig:
    """Configuration for the semantic intent cache (ChromaDB)."""
    enabled: bool = True
    path: str = "~/.friday/intent_cache"
    similarity_threshold: float = 0.75
    collection_name: str = "intents"


@dataclass
class ConversationConfig:
    """Multi-turn agent framework (see docs/multi-turn-agent-spec.md)."""
    # Layer A — short-term context register (turnstile-ctx)
    context_enabled: bool = True
    context_persist_path: str = "~/.friday/context_register.json"
    context_max_turns: int = 3
    # Layer B — durable multi-turn task sessions
    sessions_enabled: bool = True
    session_store_path: str = "~/.friday/sessions.db"
    default_session_timeout_s: int = 600
    # Background runner (drives WAITING sessions + expiry sweep)
    background_tick_seconds: int = 30


@dataclass
class ResearchConfig:
    """Learn-from-every-conversation research substrate (see research/).

    Everything is off unless `enabled` is True (set via FRIDAY_RESEARCH=1 in
    main.py, never in ephemeral --chat/--test modes). Production behavior is
    bit-identical when disabled.
    """
    enabled: bool = False
    db_path: str = "~/.friday/research.db"
    artifacts_dir: str = "~/.friday/research"
    feedback_buttons: bool = True   # 👍/👎 inline buttons on Telegram chat replies
    shadow_enabled: bool = True     # local model answers silently in parallel
    shadow_model: str = "llama3.1"
    # Which retrieval the memory arm uses during eval: existing 'facts' store
    # (baseline) or the new 'reflection' memory.
    memory_arm_retrieval: str = "facts"


@dataclass
class AssistantConfig:
    """Master configuration combining all settings."""
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    intent_cache: IntentCacheConfig = field(default_factory=IntentCacheConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    # General settings
    debug_mode: bool = False
    log_conversations: bool = True
    log_file_path: str = "conversations.log"


# Default configuration instance
# Modify this or create your own instance with custom settings
DEFAULT_CONFIG = AssistantConfig(
    personality=PersonalityConfig(
        name="Jarvis",
        user_title="sir",
        sarcasm_level=SarcasmLevel.MODERATE,
        formality_level=FormalityLevel.BUTLER,
        warmth_level=WarmthLevel.WARM,
        wit_enabled=True,
        self_aware_ai_jokes=True,
        use_british_vocabulary=True,
    ),
    tts=TTSConfig(
        provider="piper",
    ),
    stt=STTConfig(
        provider="whisper",
        whisper_model="base",
    ),
    llm=LLMConfig(
        provider="anthropic",
        anthropic_model="claude-haiku-4-5-20251001",
    ),
)
