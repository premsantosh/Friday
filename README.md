# 🤖 Jarvis Voice Assistant

A modular, extensible AI voice assistant with personality — inspired by Iron Man's Jarvis.

## Features

- **🎤 Voice Interaction**: Wake word detection → Speech-to-Text → LLM → Text-to-Speech
- **🎭 Configurable Personality**: Adjustable sarcasm, wit, formality, and warmth levels
- **🔌 Pluggable Architecture**: Easily swap TTS, STT, and LLM providers
- **🏠 Smart Home Ready**: Extensible workflow system for home automation
- **🌐 Cross-Platform**: Works on macOS, Linux, and Windows

## Quick Start

### 1. Install Dependencies

```bash
cd jarvis-assistant
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file or export environment variables:

```bash
# Required
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For ElevenLabs TTS (recommended)
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"

# For wake word detection (free at picovoice.ai)
export PORCUPINE_ACCESS_KEY="your-porcupine-access-key"

# Optional - Home Assistant integration
export HASS_URL="http://homeassistant.local:8123"
export HASS_TOKEN="your-long-lived-access-token"
```

### 3. Run the Assistant

```bash
# Normal operation (wake word + voice)
python main.py

# Debug mode (shows processing details)
python main.py --debug

# Keyboard activation (no wake word needed)
python main.py --keyboard

# Test with text (no voice required)
python main.py --test "What's the weather like?"
```

## Configuration

### Personality Settings

Edit `config/settings.py` or modify at runtime:

```python
from config import PersonalityConfig, SarcasmLevel, FormalityLevel

personality = PersonalityConfig(
    name="Jarvis",
    user_title="sir",
    sarcasm_level=SarcasmLevel.MODERATE,  # NONE, LIGHT, MODERATE, HEAVY, MAXIMUM
    formality_level=FormalityLevel.BUTLER,  # CASUAL, FRIENDLY, PROFESSIONAL, FORMAL, BUTLER
    warmth_level=WarmthLevel.WARM,  # COLD, NEUTRAL, WARM, AFFECTIONATE
    wit_enabled=True,
    self_aware_ai_jokes=True,
    use_british_vocabulary=True,
)
```

### Switching Providers

**TTS Providers:**
```python
from config import TTSConfig

# ElevenLabs (best quality)
tts_config = TTSConfig(provider="elevenlabs")

# OpenAI TTS
tts_config = TTSConfig(provider="openai")

# Piper (local, free)
tts_config = TTSConfig(provider="piper")

# System TTS (no setup required)
tts_config = TTSConfig(provider="system")
```

**STT Providers:**
```python
from config import STTConfig

# Whisper (local)
stt_config = STTConfig(provider="whisper", whisper_model="base")

# Whisper API (cloud)
stt_config = STTConfig(provider="whisper_api")

# Vosk (local, lightweight)
stt_config = STTConfig(provider="vosk")

# Deepgram (cloud, fast)
stt_config = STTConfig(provider="deepgram")
```

**LLM Providers:**
```python
from config import LLMConfig

# Anthropic Claude (recommended)
llm_config = LLMConfig(provider="anthropic")

# OpenAI GPT
llm_config = LLMConfig(provider="openai")

# Ollama (local)
llm_config = LLMConfig(provider="ollama", ollama_model="llama3.1")
```

## Adding Custom Workflows

The workflow system allows you to add new capabilities. Here's how to add a custom doorbell integration:

### 1. Create a New Workflow

```python
# workflows/my_doorbell.py
from workflows.base import Workflow, WorkflowResult, WorkflowStatus, WorkflowTrigger

class MyDoorbellWorkflow(Workflow):
    def __init__(self, doorbell_api):
        self.api = doorbell_api
    
    @property
    def name(self) -> str:
        return "my_doorbell"
    
    @property
    def description(self) -> str:
        return "Check doorbell camera and control door lock"
    
    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["door", "doorbell", "visitor", "lock"],
            patterns=[r"who.*(at|the) door", r"(lock|unlock)"],
            examples=["Who's at the door?", "Lock the front door"]
        )
    
    async def execute(self, intent: str, entities: dict) -> WorkflowResult:
        action = entities.get("action", "check")
        
        if action == "check":
            # Call your doorbell API
            snapshot = await self.api.get_snapshot()
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message="I'm checking the door camera now, sir.",
                data={"snapshot": snapshot}
            )
        
        elif action == "unlock":
            await self.api.unlock_door()
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message="I've unlocked the door, sir. Do exercise caution."
            )
        
        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message="Door action completed, sir."
        )
```

### 2. Register the Workflow

```python
# In main.py or your setup code
from workflows.my_doorbell import MyDoorbellWorkflow

workflow_manager = create_default_workflow_manager()
workflow_manager.register(MyDoorbellWorkflow(my_doorbell_api))

assistant = VoiceAssistant(config, workflow_manager)
```

## Project Structure

```
jarvis-assistant/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── config/
│   ├── __init__.py
│   └── settings.py         # All configuration dataclasses
├── core/
│   ├── __init__.py
│   └── assistant.py        # Main VoiceAssistant class
├── tts/
│   ├── __init__.py
│   └── providers.py        # TTS provider implementations
├── stt/
│   ├── __init__.py
│   └── providers.py        # STT provider implementations
├── llm/
│   ├── __init__.py
│   └── providers.py        # LLM providers + personality prompts
├── workflows/
│   ├── __init__.py
│   ├── base.py             # Workflow base classes + examples
│   └── home_assistant.py   # Home Assistant integration
└── utils/
    ├── __init__.py
    ├── audio.py            # Audio recording/playback
    └── wakeword.py         # Wake word detection
```

## Hardware Recommendations

### Microphones
- **Best**: ReSpeaker USB Mic Array v2.0 (~$70) - far-field, LED ring
- **Budget**: Anker PowerConf S3 (~$80) - mic + speaker combo
- **Testing**: Any USB microphone

### Speakers
- **Best**: Audioengine A2+ (~$270) - excellent quality
- **Budget**: Creative Pebble V3 (~$30) - USB powered

### Processing
- Works on any modern computer
- Mac Mini M4 recommended for always-on use
- Raspberry Pi 5 works for cloud-based LLM

## Troubleshooting

### "No module named 'sounddevice'"
```bash
pip install sounddevice
# On Linux, you may also need:
sudo apt-get install libportaudio2
```

### Wake word not working
1. Check `PORCUPINE_ACCESS_KEY` is set
2. Use `--keyboard` flag to test without wake word
3. Verify microphone with `python main.py --list-devices`

### TTS not working
1. Check `ELEVENLABS_API_KEY` is set
2. Falls back to system TTS automatically
3. Test with `python main.py --test "Hello world"`

### Audio crackling/choppy
- Increase buffer size in `utils/audio.py`
- Try different sample rates
- Check CPU usage

## License

MIT License - feel free to use and modify for personal projects.

## Contributing

This is a personal project template. Feel free to fork and customize!

---

*"At your service, sir."* — Jarvis
