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
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from core import VoiceAssistant, create_assistant
from workflows import (
    WorkflowManager,
    create_default_workflow_manager,
    HomeAssistantLightsWorkflow,
    HomeAssistantLockWorkflow,
    HomeAssistantClimateWorkflow,
    PhilipsHueLightsWorkflow,
)


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


def create_custom_config(args) -> AssistantConfig:
    """Create configuration based on command line arguments."""
    
    # Determine wake word provider
    wake_provider = "keyboard" if args.keyboard else "porcupine"
    
    # Determine TTS provider - default to piper (local, fast)
    # Switch to "elevenlabs" or "openai" here for cloud TTS
    tts_provider = "piper"
    
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
        ),
        wake_word=WakeWordConfig(
            provider=wake_provider,
            porcupine_keyword="jarvis",
            porcupine_sensitivity=0.5,
        ),
        debug_mode=args.debug,
    )


def create_workflow_manager() -> WorkflowManager:
    """
    Create workflow manager with all available integrations.
    
    Customize this function to add your own workflows!
    """
    manager = create_default_workflow_manager()
    
    # Add Philips Hue workflow if configured
    if os.getenv("HUE_BRIDGE_IP"):
        print("ℹ️  Philips Hue integration enabled")
        manager.unregister("lights")
        manager.register(PhilipsHueLightsWorkflow())

    # Add Home Assistant workflows if configured
    if os.getenv("HASS_TOKEN"):
        print("ℹ️  Home Assistant integration enabled")

        # Replace default light workflow with Home Assistant version
        if "hue_lights" in manager.workflows:
            manager.unregister("hue_lights")
        elif "lights" in manager.workflows:
            manager.unregister("lights")
        manager.register(HomeAssistantLightsWorkflow())

        # Add lock and climate control
        manager.register(HomeAssistantLockWorkflow())
        manager.register(HomeAssistantClimateWorkflow())

    return manager


def run_text_test(text: str, config: AssistantConfig):
    """Run a text-based test without voice."""
    print(f"\n📝 Testing with: \"{text}\"")
    print("-" * 40)

    workflow_manager = create_workflow_manager()
    assistant = VoiceAssistant(config, workflow_manager)

    response = assistant.run_single_interaction(text)
    print(f"\n🤖 {config.personality.name}: {response}")
    print()


def run_text_chat(config: AssistantConfig):
    """Run interactive chat mode - type input, voice output."""
    workflow_manager = create_workflow_manager()
    assistant = VoiceAssistant(config, workflow_manager)

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
    
    # Optional: Add callbacks for UI integration
    if args.debug:
        assistant.on_transcript = lambda t: print(f"📢 You: {t}")
        assistant.on_response = lambda r: print(f"🤖 Jarvis: {r}")
        assistant.on_error = lambda e: print(f"❌ Error: {e}")
    
    # Run!
    assistant.run()


if __name__ == "__main__":
    main()
