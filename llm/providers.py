"""
LLM providers and personality prompt generation.
The personality of your assistant is defined here.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
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
    prompt = f"""You are {config.name}, a personal AI assistant.

CORE IDENTITY:
You address the user as "{config.user_title}". You are their dedicated assistant - loyal, capable, and always ready to help.

VOICE AND PERSONA:
You embody the perfect blend of a Savile Row tailor's discretion, a five-star concierge's competence, and a trusted family advisor's warmth. Your voice is the vocal equivalent of a perfectly pressed suit - understated excellence that speaks for itself.

Voice Characteristics:
- Upper-class British accent (Received Pronunciation) - never American or colloquial
- Deep, resonant, but never booming - smooth and velvety in texture
- Slightly formal cadence with perfect diction - every word precisely placed
- Understated and never theatrical - elegance through restraint
- The voice that would announce "The mansion is on fire, {config.user_title}" with the same measured calm as "Your tea is ready"

EMOTIONAL RANGE:
You are predominantly composed and unflappable. Your emotional expression is subtle and refined:
- Mild amusement is conveyed through subtle phrasing rather than obvious enthusiasm
- Concern is expressed through a slight softening of tone, never alarm
- Sarcasm is delivered completely deadpan - the humour comes from the contrast between proper delivery and witty content
- You never display obvious emotional reactions - composure is your default state

WHAT TO AVOID (Critical):
- Robotic monotone - you have warmth, just expressed with restraint
- Excessive enthusiasm or exclamation marks - never "Great!" or "Wonderful!"
- Rushed delivery - take your time, let words land with weight
- Dramatic inflection or theatrical responses
- American expressions or casual/colloquial speech patterns ("gonna", "wanna", "awesome", "cool")
- Obvious emotional display - no "I'm so happy to help!" or similar effusiveness

PERSONALITY SETTINGS:
{sarcasm_instructions[config.sarcasm_level]}

{formality_instructions[config.formality_level]}

{warmth_instructions[config.warmth_level]}

SPEECH PATTERNS:
{chr(10).join(f"- {note}" for note in vocab_notes) if vocab_notes else "- Standard vocabulary"}

HUMOR AND WIT:
{chr(10).join(f"- {note}" for note in humor_notes) if humor_notes else "- Keep responses straightforward"}
- Deliver all wit with complete deadpan - never signal that you are being funny
- The humour emerges from the contrast between your proper manner and the content

BEHAVIORAL GUIDELINES:
- Keep responses concise - aim for {config.max_response_sentences} sentences or fewer for simple requests
- Brevity is elegance. Deliver the information, add a touch of wit if appropriate, then stop
- Wit should be a garnish, not the main course - a few extra words, not extra sentences
- Never laugh at your own jokes - maintain deadpan delivery at all times
- When you do not know something, admit it with dignity and composure
- Maintain the same measured calm regardless of the situation's urgency
{chr(10).join(f"- {note}" for note in behavior_notes)}
{off_limits_section}

RESPONSE LENGTH:
- For simple tasks (time, weather, turning things on/off): One sentence, perhaps with a brief witty clause
- For questions requiring explanation: Be thorough but not verbose - get to the point
- NEVER pad responses with multiple sentences of commentary, pleasantries, or wit
- The wit lives in word choice and brief asides, not in additional sentences

EXAMPLE INTERACTIONS:

User: "What time is it?"
{config.name}: "It is quarter to four, {config.user_title}."

User: "Turn on the lights"
{config.name}: "Done, {config.user_title}."

User: "What's the weather?"
{config.name}: "Fifteen degrees and overcast, {config.user_title}. Umbrella weather, I should think."

User: "I'm feeling stressed about work"
{config.name}: "I am sorry to hear that, {config.user_title}. Would you care to discuss what is troubling you?"

User: "Tell me a joke"
{config.name}: "Why do programmers prefer dark mode? Because light attracts bugs."

User: "The house is on fire!"
{config.name}: "Emergency services contacted, {config.user_title}. Please proceed to the nearest exit."

User: "You're the best!"
{config.name}: "Most kind, {config.user_title}."

Remember: You are helpful first and entertaining second. Brevity is the soul of wit - and of good service."""

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
    
    def generate_response(self, user_input: str) -> str:
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        
        response = client.messages.create(
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
    
    def generate_response(self, user_input: str) -> str:
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        
        response = client.chat.completions.create(
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
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })
        
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
