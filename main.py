#!/usr/bin/env python3
"""
Jarvis Voice Assistant - Main Entry Point

Usage:
    python main.py                    # Run with default settings
    python main.py --debug            # Run with debug output
    python main.py --keyboard         # Use keyboard activation (no wake word)
    python main.py --chat             # Interactive text mode (no microphone)
    python main.py --test "Hello"     # Test with single text input (no voice)

Environment Variables:
    ANTHROPIC_API_KEY     - Required for Claude LLM
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
from core import VoiceAssistant, create_assistant
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


def check_api_keys():
    """Check for required API keys and warn if missing."""
    warnings = []
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        warnings.append("ANTHROPIC_API_KEY not set - Claude LLM won't work")
    
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
            provider="anthropic",
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


def run_text_chat(config: AssistantConfig):
    """Run interactive chat mode - type input, voice output."""
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

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                assistant.speak(f"Very good, {config.personality.user_title}. Until next time.")
                break

            if user_input.lower() == "clear":
                assistant.clear_history()
                print("[Conversation history cleared]\n")
                continue

            response = assistant.run_single_interaction(user_input)
            print(f"\n🤖 {name}: {response}\n")
            assistant.speak(response)

        except KeyboardInterrupt:
            print()
            assistant.speak(f"Very good, {config.personality.user_title}. Until next time.")
            break
        except EOFError:
            print()
            break


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
        help="Interactive text chat mode (no microphone required)",
    )

    parser.add_argument(
        "--test",
        type=str,
        metavar="TEXT",
        help="Test with single text input (no voice)",
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
    
    # Check API keys
    check_api_keys()
    
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

    # Create workflow manager
    workflow_manager = create_workflow_manager()

    # Create and run assistant
    assistant = VoiceAssistant(config, workflow_manager)
    assistant.llm.search_enhancer = create_search_enhancer(config)

    # Optional: Add callbacks for UI integration
    if args.debug:
        assistant.on_transcript = lambda t: print(f"📢 You: {t}")
        assistant.on_response = lambda r: print(f"🤖 Jarvis: {r}")
        assistant.on_error = lambda e: print(f"❌ Error: {e}")

    # Start background monitors that need a speak callback
    coffee_workflow = workflow_manager.get_workflow("coffee_machine")
    if coffee_workflow is not None:
        coffee_workflow.start_monitor(assistant.speak)

    # Run!
    assistant.run()

    # Clean up background monitors on exit
    if coffee_workflow is not None:
        coffee_workflow.stop_monitor()


if __name__ == "__main__":
    main()
