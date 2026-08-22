"""
LLMConfig -> LangChain chat model factory.

Provider choice stays configurable (LLMConfig.provider), mirroring the legacy
`get_llm_provider` registry. Only langchain-anthropic is a declared dependency;
OpenAI/Ollama bindings are imported lazily and fail soft with a clear error.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from config import AgentConfig, LLMConfig


def build_chat_model(llm: LLMConfig, agent: AgentConfig) -> BaseChatModel:
    provider = llm.provider
    temperature = llm.temperature
    max_tokens = agent.max_tokens

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = llm.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        return ChatAnthropic(
            model=agent.model or llm.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LLM provider 'openai' needs `pip install langchain-openai`") from exc
        api_key = llm.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return ChatOpenAI(
            model=agent.model or llm.openai_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LLM provider 'ollama' needs `pip install langchain-ollama`") from exc
        # Tool support varies by Ollama model; the engine fails soft to legacy
        # if the model rejects bound tools.
        return ChatOllama(
            model=agent.model or llm.ollama_model,
            base_url=llm.ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )

    raise ValueError(f"Unknown LLM provider for the agent engine: {provider}")


def model_label(llm: LLMConfig, agent: AgentConfig) -> Optional[str]:
    """Short label for startup banners."""
    name = agent.model or {
        "anthropic": llm.anthropic_model,
        "openai": llm.openai_model,
        "ollama": llm.ollama_model,
    }.get(llm.provider)
    return f"{llm.provider} ({name})" if name else None
