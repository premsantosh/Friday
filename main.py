#!/usr/bin/env python3
"""
Jarvis Voice Assistant - Main Entry Point

Usage:
    python main.py                    # Listen on voice + text + Telegram together
    python main.py --debug            # Run with debug output
    python main.py --keyboard         # Keyboard activation for voice (+ Telegram)
    python main.py --chat             # Text only (no microphone, no Telegram)
    python main.py --telegram         # Headless Telegram bot (no mic/terminal)
    python main.py --test "Hello"     # Test with single text input (no voice)

By default Friday listens on every channel that's available: voice (wake word),
text (type in the terminal), and Telegram (when TELEGRAM_BOT_TOKEN is
configured). No flags are needed to enable them.

Privacy: conversations are processed on-device by default (Ollama). Set
FRIDAY_LLM_PROVIDER=anthropic (or openai) to explicitly opt in to a cloud
LLM — even then, private memory (preferences, personal facts, summaries,
device state) is never included in prompts that leave the device.

Environment Variables:
    FRIDAY_LLM_PROVIDER   - LLM provider: ollama (default, local), anthropic, openai
    FRIDAY_OLLAMA_MODEL   - Local conversation model (default: llama3.1)
    ANTHROPIC_API_KEY     - Required only if FRIDAY_LLM_PROVIDER=anthropic
    ELEVENLABS_API_KEY    - Required for ElevenLabs TTS
    PORCUPINE_ACCESS_KEY  - Required for Porcupine wake word (free at picovoice.ai)
    OPENAI_API_KEY        - Optional, for OpenAI TTS/LLM/Whisper API
    HASS_URL              - Optional, Home Assistant URL
    HASS_TOKEN            - Optional, Home Assistant access token
    HUE_BRIDGE_IP         - Optional, Philips Hue Bridge IP address
    HUE_APPLICATION_KEY   - Optional, Philips Hue API application key
    SHELLY_DEVICES        - Optional, JSON list of Shelly devices, e.g.:
                            '[{"name":"desk lamp","ip":"192.168.1.50","type":"plug"}]'
    COFFEE_MACHINE_IP     - Optional, IP of the Shelly plug on the coffee machine
    COFFEE_MACHINE_WEEKDAY_ON - HH:MM auto-on time on weekdays (default: 07:00)
    COFFEE_MACHINE_WEEKEND_ON - HH:MM auto-on time on weekends (default: 09:00)
    COFFEE_MACHINE_AUTO_ON    - true/false, enable scheduled auto-on (default: true)
    COFFEE_MACHINE_IDLE_ALERT_MIN - minutes idle before alert (default: 45)
    COFFEE_MACHINE_WARMUP_MINUTES - warm-up time in minutes (default: 20)
"""

import argparse
import os
import sys
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load ./.env before anything reads os.getenv, so configuration works regardless
# of shell (zsh doesn't source ~/.bash_profile). Shell vars already set win over
# the file (override=False), and a missing .env is a harmless no-op.
from dotenv import load_dotenv
load_dotenv()

from config import (
    AssistantConfig,
    PersonalityConfig,
    TTSConfig,
    STTConfig,
    LLMConfig,
    WakeWordConfig,
    IntentCacheConfig,
    SarcasmLevel,
    FormalityLevel,
    WarmthLevel,
)
from core import VoiceAssistant, create_assistant, TelegramChannel
from workflows import (
    WorkflowManager,
    create_default_workflow_manager,
    HomeAssistantLockWorkflow,
    HomeAssistantClimateWorkflow,
    PhilipsHueLightsWorkflow,
    TimeWorkflow,
    ShellyWorkflow,
    CoffeeMachineWorkflow,
)
from search import OllamaSearchClassifier, TavilySearchProvider, SearchEnhancer


def _ollama_reachable(base_url: str = "http://localhost:11434") -> bool:
    import requests
    try:
        requests.get(f"{base_url}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def check_api_keys(llm_provider: str = "ollama"):
    """Check for required API keys / local services and warn if missing."""
    warnings = []

    if llm_provider == "ollama":
        if not _ollama_reachable():
            warnings.append(
                "Ollama is not reachable at localhost:11434 - the local LLM won't work.\n"
                "     Start it with 'ollama serve' and pull a model ('ollama pull llama3.1').\n"
                "     To use a cloud LLM instead (conversations leave the device; private\n"
                "     memory is still withheld), set FRIDAY_LLM_PROVIDER=anthropic."
            )
    elif llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        warnings.append("ANTHROPIC_API_KEY not set - Claude LLM won't work")
    elif llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set - OpenAI LLM won't work")

    if not os.getenv("ELEVENLABS_API_KEY"):
        warnings.append("ELEVENLABS_API_KEY not set - ElevenLabs TTS unavailable (using Piper by default)")
    
    if not os.getenv("PORCUPINE_ACCESS_KEY"):
        warnings.append("PORCUPINE_ACCESS_KEY not set - wake word detection disabled")

    if os.getenv("HUE_BRIDGE_IP") and not (os.getenv("HUE_APPLICATION_KEY") or os.getenv("HUE_USERNAME")):
        warnings.append("HUE_BRIDGE_IP is set but HUE_APPLICATION_KEY is missing - Philips Hue integration won't work")

    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for w in warnings:
            print(f"   • {w}")
        print()
    
    return len(warnings) == 0


def _build_home_context() -> str:
    """
    Assemble a plain-text knowledge block about configured home appliances.
    This is injected into the LLM system prompt so the assistant knows
    device-specific facts even in general conversation (not just workflows).
    """
    parts = []

    if os.getenv("COFFEE_MACHINE_IP"):
        warmup = int(os.getenv("COFFEE_MACHINE_WARMUP_MINUTES", "20"))
        wd_on = os.getenv("COFFEE_MACHINE_WEEKDAY_ON", "07:00")
        we_on = os.getenv("COFFEE_MACHINE_WEEKEND_ON", "09:00")
        name = os.getenv("COFFEE_MACHINE_NAME", "coffee machine")
        parts.append(
            f"Coffee machine ({name}):\n"
            f"- Takes approximately {warmup} minutes to heat up from cold and be ready to brew.\n"
            f"  Always mention this warm-up time when starting the machine.\n"
            f"- Auto-start schedule: {wd_on} on weekdays, {we_on} on weekends.\n"
            f"- Any mention of a coffee drink (espresso, latte, cappuccino, americano,\n"
            f"  macchiato, mocha, flat white, cortado, lungo, etc.) — whether from the user\n"
            f"  or on behalf of guests — is an implicit request to start the machine.\n"
            f"- If asked 'can I have coffee in X minutes?', compare X to the {warmup}-min warm-up\n"
            f"  and answer honestly."
        )

    return "\n\n".join(parts)


def create_custom_config(args) -> AssistantConfig:
    """Create configuration based on command line arguments."""

    # Determine wake word provider
    wake_provider = "keyboard" if args.keyboard else "porcupine"

    # Determine TTS provider — ElevenLabs needs both its key and a voice ID;
    # otherwise fall back to the OS voice so text modes still run.
    if os.getenv("ELEVENLABS_API_KEY") and os.getenv("ELEVENLABS_VOICE_ID"):
        tts_provider = "elevenlabs"
    else:
        tts_provider = "system"

    is_ephemeral = bool(args.chat or args.test)

    return AssistantConfig(
        personality=PersonalityConfig(
            name="Jarvis",
            user_title="sir",
            sarcasm_level=SarcasmLevel.MODERATE,
            formality_level=FormalityLevel.BUTLER,
            warmth_level=WarmthLevel.WARM,
            wit_enabled=True,
            self_aware_ai_jokes=True,
            observational_humor=True,
            use_british_vocabulary=True,
            use_contractions=False,
            home_context=_build_home_context(),
        ),
        tts=TTSConfig(
            provider=tts_provider,
        ),
        stt=STTConfig(
            provider="whisper",
            whisper_model="base",  # Use "small" or "medium" for better accuracy
        ),
        llm=LLMConfig(
            # Local-first: conversations stay on-device unless the owner
            # explicitly opts in to a cloud provider via FRIDAY_LLM_PROVIDER.
            provider=os.getenv("FRIDAY_LLM_PROVIDER", "ollama"),
            ollama_model=os.getenv("FRIDAY_OLLAMA_MODEL", "llama3.1"),
            anthropic_model="claude-haiku-4-5-20251001",
            ephemeral=is_ephemeral,
        ),
        wake_word=WakeWordConfig(
            provider=wake_provider,
            porcupine_keyword="jarvis",
            porcupine_sensitivity=0.5,
        ),
        intent_cache=IntentCacheConfig(enabled=not is_ephemeral),
        debug_mode=args.debug,
    )


def create_workflow_manager() -> WorkflowManager:
    """
    Create workflow manager with all available integrations.

    Customize this function to add your own workflows!
    """
    manager = create_default_workflow_manager()

    # Time workflow - always available
    manager.register(TimeWorkflow())
    
    # Add Philips Hue workflow if configured
    if os.getenv("HUE_BRIDGE_IP"):
        print("ℹ️  Philips Hue integration enabled")
        manager.register(PhilipsHueLightsWorkflow())

    # Add Home Assistant workflows if configured
    if os.getenv("HASS_TOKEN"):
        print("ℹ️  Home Assistant integration enabled")
        manager.register(HomeAssistantLockWorkflow())
        manager.register(HomeAssistantClimateWorkflow())

    # Add Shelly smart plug workflow if devices are configured
    if os.getenv("SHELLY_DEVICES"):
        print("ℹ️  Shelly integration enabled")
        manager.register(ShellyWorkflow())

    # Add Coffee Machine workflow if configured
    if os.getenv("COFFEE_MACHINE_IP"):
        print("ℹ️  Coffee machine integration enabled")
        manager.register(CoffeeMachineWorkflow())

    # Add Reservations agent if enabled
    if os.getenv("RESERVATIONS_ENABLED", "").lower() in ("1", "true", "yes"):
        print("ℹ️  Reservations agent enabled")
        from workflows.reservations import ReservationWorkflow
        manager.register(ReservationWorkflow())

    return manager


def create_search_enhancer(config: AssistantConfig):
    """Create and return a SearchEnhancer, or None if search is disabled/unconfigured."""
    if not config.search.enabled:
        return None
    api_key = config.search.tavily_api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    classifier = OllamaSearchClassifier(
        base_url=config.llm.ollama_base_url,
        model=config.search.classifier_model,
    )
    provider = TavilySearchProvider(api_key=api_key)
    return SearchEnhancer(classifier, provider, max_results=config.search.max_results)


def run_text_test(text: str, config: AssistantConfig):
    """Run a text-based test without voice."""
    print(f"\n📝 Testing with: \"{text}\"")
    print("-" * 40)

    workflow_manager = create_workflow_manager()
    assistant = VoiceAssistant(config, workflow_manager)
    assistant.llm.search_enhancer = create_search_enhancer(config)

    response = assistant.run_single_interaction(text)
    print(f"\n🤖 {config.personality.name}: {response}")
    print()


def _text_chat_loop(assistant: VoiceAssistant, config: AssistantConfig) -> str:
    """Foreground "type a message" loop, shared by chat mode and the default
    all-channels mode.

    Returns why it ended: 'quit' (user typed quit/exit), 'interrupt' (Ctrl+C),
    or 'eof' (Ctrl+D / no stdin). Callers running other channels concurrently
    use that to decide whether to keep those channels alive.
    """
    name = config.personality.name
    title = config.personality.user_title
    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print()
            assistant.speak(f"Very good, {title}. Until next time.")
            return "interrupt"
        except EOFError:
            print()
            return "eof"

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            assistant.speak(f"Very good, {title}. Until next time.")
            return "quit"

        if user_input.lower() == "clear":
            assistant.clear_history()
            print("[Conversation history cleared]\n")
            continue

        response = assistant.run_single_interaction(user_input)
        print(f"\n🤖 {name}: {response}\n")
        assistant.speak(response)


def run_text_chat(config: AssistantConfig):
    """Run interactive chat mode - type input, voice output (text only, no mic)."""
    workflow_manager = create_workflow_manager()
    assistant = VoiceAssistant(config, workflow_manager)
    assistant.llm.search_enhancer = create_search_enhancer(config)

    name = config.personality.name
    print(f"\n{'='*50}")
    print(f"  {name} - Chat Mode")
    print(f"{'='*50}")
    print(f"  TTS: {assistant.tts.get_name()}")
    print(f"  LLM: {assistant.llm.get_name()}")
    print(f"{'='*50}")
    print(f"Type your messages below. Commands:")
    print(f"  'quit' or 'exit' - End the conversation")
    print(f"  'clear'          - Clear conversation history")
    print(f"{'='*50}\n")

    _text_chat_loop(assistant, config)


def _attach_listener(assistant: VoiceAssistant, channel) -> None:
    """Wire a messaging channel's inbound messages to the assistant brain.

    Works for any channel exposing start(handler) where handler is
    `async (text, sender) -> reply`. Each sender gets their own multi-turn
    session via user_id, so a chat tracks reservations/dialogues independently
    of the voice path. (Telegram keys by chat ID.)
    """
    async def handle(text: str, sender: str) -> Optional[str]:
        return await assistant.process_input(text, user_id=sender)

    channel.start(handle)


def _run_headless_channel(config: AssistantConfig, channel, label: str,
                          owner, summary: str):
    """Run Friday headless against a single messaging channel — no mic/speaker.

    You text Friday; it routes the message through the same brain the voice
    assistant uses and texts the reply back. Long-running outcomes (reservation
    results, coffee alerts) are pushed to `owner` instead of being spoken aloud.
    """
    workflow_manager = create_workflow_manager()
    assistant = VoiceAssistant(config, workflow_manager)
    assistant.llm.search_enhancer = create_search_enhancer(config)

    notify_owner = (lambda msg: channel.send(msg, owner)) if owner else (lambda msg: None)

    # Re-point session notifications at the channel, since there's no speaker
    # in this mode (the assistant built its runner wired to speak()).
    if assistant.sessions is not None:
        from core.conversation import BackgroundTaskRunner
        assistant.background_runner = BackgroundTaskRunner(
            assistant.sessions,
            notify_owner,
            tick_seconds=config.conversation.background_tick_seconds,
        )
        assistant.background_runner.start()

    # Coffee-machine alerts also have nowhere to speak — route them to the channel.
    coffee_workflow = workflow_manager.get_workflow("coffee_machine")
    if coffee_workflow is not None and owner:
        coffee_workflow.start_monitor(notify_owner)

    _attach_listener(assistant, channel)

    name = config.personality.name
    print(f"\n{'='*50}")
    print(f"  {name} - {label} Mode")
    print(f"{'='*50}")
    print(f"  LLM: {assistant.llm.get_name()}")
    print(f"  {summary}")
    print(f"{'='*50}")
    print("  Text Friday from your phone. (Press Ctrl+C to quit)\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        channel.stop()
        if assistant.background_runner is not None:
            assistant.background_runner.stop()
        if coffee_workflow is not None:
            coffee_workflow.stop_monitor()


def run_telegram_mode(config: AssistantConfig):
    """Run Friday headless as a Telegram bot — no mic, no speaker."""
    channel = TelegramChannel.from_env()
    if channel is None:
        print("\n❌ Telegram channel not configured.")
        print("   Set TELEGRAM_BOT_TOKEN (from @BotFather). Optionally set")
        print("   TELEGRAM_ALLOWED_CHAT_IDS to authorise chats.\n")
        return
    owner = next(iter(channel.allowed_chat_ids), None)
    summary = (f"Authorised chats: {', '.join(sorted(channel.allowed_chat_ids))}"
               if channel.allowed_chat_ids
               else "No allowlist yet — message the bot and it will reply with your chat ID.")
    _run_headless_channel(config, channel, "Telegram", owner, summary)


def run_all(config: AssistantConfig, debug: bool = False):
    """Default mode: listen on voice, text, and Telegram at once.

    Voice (wake word, mic) and Telegram (inbound poll) run on background threads;
    the type-a-message loop owns the foreground. Any channel that isn't available
    is skipped silently — no flags required to opt in/out:
      * Voice needs a working wake-word detector (mic).
      * Telegram needs TELEGRAM_BOT_TOKEN configured.
      * Text is always available when there's a terminal.
    """
    workflow_manager = create_workflow_manager()
    assistant = VoiceAssistant(config, workflow_manager)
    assistant.llm.search_enhancer = create_search_enhancer(config)

    if debug:
        assistant.on_transcript = lambda t: print(f"📢 You: {t}")
        assistant.on_response = lambda r: print(f"🤖 {config.personality.name}: {r}")
        assistant.on_error = lambda e: print(f"❌ Error: {e}")

    name = config.personality.name
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  TTS: {assistant.tts.get_name()}")
    print(f"  LLM: {assistant.llm.get_name()}")

    # Coffee-machine alerts speak aloud (mic/speaker present in this mode).
    coffee_workflow = workflow_manager.get_workflow("coffee_machine")
    if coffee_workflow is not None:
        coffee_workflow.start_monitor(assistant.speak)

    # Telegram — automatically on whenever it's configured.
    telegram_channel = TelegramChannel.from_env()
    if telegram_channel is not None:
        _attach_listener(assistant, telegram_channel)

    # Voice — start in the background; don't grab stdin (the text loop owns it),
    # so a mic-less machine just runs text + Telegram instead of hijacking input.
    voice_active = assistant.start_listening(allow_keyboard_fallback=False)

    channels = []
    if voice_active:
        channels.append(f"voice (say '{config.wake_word.porcupine_keyword}')")
    channels.append("text (type below)")
    if telegram_channel is not None:
        channels.append("Telegram")
    print(f"  Channels: {' · '.join(channels)}")
    print(f"{'='*50}")
    print("  Talk, type, or text Friday. (Ctrl+C to quit)\n")

    try:
        reason = _text_chat_loop(assistant, config)
        # Ctrl+D / closed stdin shouldn't kill voice + Telegram — keep them alive.
        if reason == "eof" and (voice_active or telegram_channel is not None):
            print("[text input closed — voice/Telegram still listening; Ctrl+C to quit]")
            while assistant._running:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        assistant.stop()
        if coffee_workflow is not None:
            coffee_workflow.stop_monitor()
        if telegram_channel is not None:
            telegram_channel.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Jarvis Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Use keyboard activation instead of wake word",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Text-only chat mode (no microphone, no Telegram)",
    )

    parser.add_argument(
        "--test",
        type=str,
        metavar="TEXT",
        help="Test with single text input (no voice)",
    )

    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Run headless as a Telegram bot (chat with Friday from your phone)",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    
    args = parser.parse_args()

    # PII/card redaction over everything the logging module emits (harness §5.2).
    from core.harness import install_log_redaction
    install_log_redaction()

    # List audio devices
    if args.list_devices:
        from utils import list_audio_devices
        list_audio_devices()
        return
    
    # Check API keys / local LLM availability
    check_api_keys(os.getenv("FRIDAY_LLM_PROVIDER", "ollama"))

    # Create configuration
    config = create_custom_config(args)
    
    # Text test mode
    if args.test:
        run_text_test(args.test, config)
        return

    # Interactive text chat mode
    if args.chat:
        run_text_chat(config)
        return

    # Headless Telegram bot mode (no microphone)
    if args.telegram:
        run_telegram_mode(config)
        return

    # Keyboard-activation mode owns stdin for the wake trigger, so it can't share
    # the foreground with a text loop — run voice (keyboard) + Telegram only.
    if args.keyboard:
        workflow_manager = create_workflow_manager()
        assistant = VoiceAssistant(config, workflow_manager)
        assistant.llm.search_enhancer = create_search_enhancer(config)

        if args.debug:
            assistant.on_transcript = lambda t: print(f"📢 You: {t}")
            assistant.on_response = lambda r: print(f"🤖 {config.personality.name}: {r}")
            assistant.on_error = lambda e: print(f"❌ Error: {e}")

        coffee_workflow = workflow_manager.get_workflow("coffee_machine")
        if coffee_workflow is not None:
            coffee_workflow.start_monitor(assistant.speak)

        telegram_channel = TelegramChannel.from_env()
        if telegram_channel is not None:
            _attach_listener(assistant, telegram_channel)
            print("ℹ️  Telegram two-way channel enabled")

        assistant.run()

        if coffee_workflow is not None:
            coffee_workflow.stop_monitor()
        if telegram_channel is not None:
            telegram_channel.stop()
        return

    # Default: listen on voice + text + Telegram together.
    run_all(config, debug=args.debug)


if __name__ == "__main__":
    main()
