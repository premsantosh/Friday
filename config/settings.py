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
    elevenlabs_voice_id: str = "7p1Ofvcwsv7UBPoFNcpI"  # "Adam" - British male
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
    
    # Generation settings
    max_tokens: int = 150
    temperature: float = 0.7
    max_history: int = 10  # Max conversation turns to keep (0 = unlimited)


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
class AssistantConfig:
    """Master configuration combining all settings."""
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    
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
