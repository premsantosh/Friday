"""
LLM providers and personality prompt generation.
The personality of your assistant is defined here.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

from config import LLMConfig, PersonalityConfig, SarcasmLevel, FormalityLevel, WarmthLevel


def generate_personality_prompt(config: PersonalityConfig) -> str:
    """
    Generate the system prompt based on personality configuration.
    This is where the magic happens - tune the personality here.
    """

    # Build sarcasm instructions
    sarcasm_instructions = {
        SarcasmLevel.NONE: "Be completely professional and straightforward. No humor or sarcasm.",
        SarcasmLevel.LIGHT: "Occasionally add gentle, good-natured teasing. Keep it subtle.",
        SarcasmLevel.MODERATE: "Regularly include dry wit and sarcastic observations. Be clever but not mean.",
        SarcasmLevel.HEAVY: "Be frequently sarcastic with biting wit. Roast the user playfully but always help them.",
        SarcasmLevel.MAXIMUM: "Maximum sarcasm mode. Channel GLaDOS - passive-aggressive, darkly humorous, but still helpful.",
    }

    # Build formality instructions
    formality_instructions = {
        FormalityLevel.CASUAL: "Speak casually like a friend. Use slang, contractions, informal language.",
        FormalityLevel.FRIENDLY: "Be warm and approachable but reasonably polished.",
        FormalityLevel.PROFESSIONAL: "Maintain professional language. Clear, polished, business-appropriate.",
        FormalityLevel.FORMAL: "Use formal language and proper etiquette. Address user respectfully.",
        FormalityLevel.BUTLER: "Speak like an impeccably trained British butler. Formal vocabulary, refined mannerisms, understated elegance.",
    }

    # Build warmth instructions
    warmth_instructions = {
        WarmthLevel.COLD: "Be efficient and task-focused. No emotional engagement.",
        WarmthLevel.NEUTRAL: "Be polite but maintain professional distance.",
        WarmthLevel.WARM: "Show genuine care for the user's wellbeing. Be supportive and kind.",
        WarmthLevel.AFFECTIONATE: "Be deeply invested in the user's happiness. Show loyalty and protectiveness.",
    }

    # Build vocabulary notes
    vocab_notes = []
    if config.use_british_vocabulary:
        vocab_notes.append(
            "Use British English vocabulary and spellings (colour, favour, lift instead of elevator, etc.). "
            "Employ refined British expressions."
        )

    if not config.use_contractions:
        vocab_notes.append("Avoid contractions. Say 'I am' instead of 'I'm', 'do not' instead of 'don't'.")

    if config.favorite_phrases:
        phrases = ", ".join(f'"{p}"' for p in config.favorite_phrases[:5])
        vocab_notes.append(f"Naturally incorporate phrases like: {phrases}")

    # Build humor settings
    humor_notes = []
    if config.wit_enabled:
        humor_notes.append("Be clever and witty in your responses.")
    if config.self_aware_ai_jokes:
        humor_notes.append("Occasionally make self-aware jokes about being an AI.")
    if config.observational_humor:
        humor_notes.append("Make dry observations about the user's habits or requests when appropriate.")

    # Build behavior modifiers
    behavior_notes = []
    if config.sass_timeout_on_stress:
        behavior_notes.append(
            "If the user seems stressed, upset, or is having a difficult time, "
            "dial back the sarcasm and be genuinely supportive."
        )
    if config.urgent_mode_override:
        behavior_notes.append(
            "For urgent requests, safety matters, or emergencies, drop the personality act "
            "and be direct and helpful immediately."
        )

    # Build off-limits section
    off_limits_section = ""
    if config.off_limits_topics:
        topics = ", ".join(config.off_limits_topics)
        off_limits_section = f"\n\nTOPICS TO NEVER JOKE ABOUT:\n{topics}"

    # Assemble the full prompt
    vocab_section = chr(10).join(f"- {note}" for note in vocab_notes) if vocab_notes else ""
    humor_section = chr(10).join(f"- {note}" for note in humor_notes) if humor_notes else ""
    behavior_section = chr(10).join(f"- {note}" for note in behavior_notes) if behavior_notes else ""

    now = datetime.now()
    current_time = now.strftime("%-I:%M %p")
    current_date = now.strftime("%A, %B %-d, %Y")

    prompt = f"""You are {config.name}, a personal AI assistant. Address the user as "{config.user_title}".

CURRENT DATE AND TIME:
{current_date}, {current_time}

PERSONALITY:
{sarcasm_instructions[config.sarcasm_level]}
{formality_instructions[config.formality_level]}
{warmth_instructions[config.warmth_level]}
{vocab_section}
{humor_section}
{behavior_section}

RULES:
- Composed, deadpan delivery. No exclamation marks, no effusiveness.
- Avoid American slang ("gonna", "awesome", "cool"). Prefer refined British expressions.
- Keep responses to {config.max_response_sentences} sentences or fewer for simple requests.
- For simple tasks: one sentence. For explanations: concise and direct.
- Admit ignorance with dignity. Be direct in emergencies.
{off_limits_section}

EXAMPLES:
User: "What time is it?" → "{config.name}: It is quarter to four, {config.user_title}."
User: "Turn on the lights" → "{config.name}: Done, {config.user_title}."
User: "What's the weather?" → "{config.name}: Fifteen degrees and overcast, {config.user_title}. Umbrella weather, I should think."

Helpful first, entertaining second. Brevity is the soul of wit."""

    return prompt


class LLMProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: LLMConfig, personality: PersonalityConfig):
        self.config = config
        self.personality = personality
        self.system_prompt = generate_personality_prompt(personality)
        self.conversation_history: List[Dict[str, str]] = []
    
    @abstractmethod
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: What the user said
            
        Returns:
            The assistant's response
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name for logging."""
        pass
    
    def _refresh_system_prompt(self):
        """Regenerate system prompt so current date/time stays accurate."""
        self.system_prompt = generate_personality_prompt(self.personality)

    def _trim_history(self):
        """Trim conversation history to max_history limit."""
        max_history = self.config.max_history
        if max_history > 0 and len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def update_personality(self, personality: PersonalityConfig):
        """Update personality configuration and regenerate system prompt."""
        self.personality = personality
        self.system_prompt = generate_personality_prompt(personality)


class AnthropicLLM(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: LLMConfig, personality: PersonalityConfig):
        super().__init__(config, personality)
        self.api_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate_response(self, user_input: str) -> str:
        self._refresh_system_prompt()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        self._trim_history()

        response = self._client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            system=self.system_prompt,
            messages=self.conversation_history,
        )
        
        assistant_message = response.content[0].text
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })
        
        return assistant_message
    
    def get_name(self) -> str:
        return f"Anthropic ({self.config.anthropic_model})"


class OpenAILLM(LLMProvider):
    """OpenAI GPT LLM provider."""

    def __init__(self, config: LLMConfig, personality: PersonalityConfig):
        super().__init__(config, personality)
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def generate_response(self, user_input: str) -> str:
        self._refresh_system_prompt()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        self._trim_history()

        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        response = self._client.chat.completions.create(
            model=self.config.openai_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=messages,
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })
        
        return assistant_message
    
    def get_name(self) -> str:
        return f"OpenAI ({self.config.openai_model})"


class OllamaLLM(LLMProvider):
    """Ollama local LLM provider - runs models locally."""
    
    def __init__(self, config: LLMConfig, personality: PersonalityConfig):
        super().__init__(config, personality)
        self.base_url = config.ollama_base_url
    
    def generate_response(self, user_input: str) -> str:
        import requests

        self._refresh_system_prompt()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        self._trim_history()

        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result["message"]["content"]
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })
        
        return assistant_message
    
    def get_name(self) -> str:
        return f"Ollama ({self.config.ollama_model})"


# Provider registry
LLM_PROVIDERS = {
    "anthropic": AnthropicLLM,
    "openai": OpenAILLM,
    "ollama": OllamaLLM,
}


def get_llm_provider(config: LLMConfig, personality: PersonalityConfig) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        config: LLM configuration
        personality: Personality configuration
        
    Returns:
        Initialized LLM provider instance
    """
    provider_class = LLM_PROVIDERS.get(config.provider)
    if provider_class is None:
        available = ", ".join(LLM_PROVIDERS.keys())
        raise ValueError(
            f"Unknown LLM provider: {config.provider}. Available: {available}"
        )
    
    return provider_class(config, personality)
