"""
Intent router: uses the configured LLM to classify user input into a workflow
when keyword matching fails. Returns structured routing + a natural response.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an intent classifier for a smart home voice assistant.
Determine if the user's request maps to one of the available workflows, extract
relevant entities, and draft a natural spoken response.

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "workflow": "<workflow_name or null>",
  "entities": {},
  "response": "<one sentence spoken response>"
}

Set "workflow" to null when no workflow fits — provide a conversational response."""


@dataclass
class RouteResult:
    workflow_name: Optional[str]
    entities: dict = field(default_factory=dict)
    response: str = ""


class IntentRouter:
    """
    Calls the configured LLM provider with a minimal routing prompt (no
    personality, no conversation history) to classify intent and extract entities.
    """

    def __init__(self, llm_config):
        self.config = llm_config
        self._anthropic_client = None
        self._openai_client = None

    def route(self, text: str, workflow_manager) -> RouteResult:
        workflow_context = self._build_workflow_context(workflow_manager)
        user_message = f"{workflow_context}\n\nUser said: \"{text}\""

        try:
            raw = self._call_llm(user_message)
            return self._parse(raw)
        except Exception:
            logger.exception("Intent router failed for: %s", text)
            return RouteResult(workflow_name=None)

    def _build_workflow_context(self, workflow_manager) -> str:
        lines = ["Available workflows:"]
        for workflow in workflow_manager.workflows.values():
            examples = ", ".join(f'"{ex}"' for ex in workflow.trigger.examples[:3])
            lines.append(f"  {workflow.name}: {workflow.description}. Examples: {examples}")
        return "\n".join(lines)

    def _call_llm(self, user_message: str) -> str:
        provider = self.config.provider
        if provider == "anthropic":
            return self._call_anthropic(user_message)
        if provider == "openai":
            return self._call_openai(user_message)
        if provider == "ollama":
            return self._call_ollama(user_message)
        raise ValueError(f"Unsupported LLM provider for routing: {provider}")

    def _call_anthropic(self, user_message: str) -> str:
        if self._anthropic_client is None:
            import anthropic
            api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)

        response = self._anthropic_client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=200,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_openai(self, user_message: str) -> str:
        if self._openai_client is None:
            from openai import OpenAI
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = OpenAI(api_key=api_key)

        response = self._openai_client.chat.completions.create(
            model=self.config.openai_model,
            max_tokens=200,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _call_ollama(self, user_message: str) -> str:
        import requests

        response = requests.post(
            f"{self.config.ollama_base_url}/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 200},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _parse(self, raw: str) -> RouteResult:
        # Strip markdown code fences Claude sometimes adds (```json ... ```)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[-1] if text.count("```") >= 2 else text
            text = text.lstrip("json").strip().rstrip("```").strip()

        try:
            data = json.loads(text)
            return RouteResult(
                workflow_name=data.get("workflow") or None,
                entities=data.get("entities") or {},
                response=data.get("response", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            # Don't speak raw JSON/text — return empty so caller falls back to LLM
            logger.warning("Router returned unparseable response: %s", raw[:200])
            return RouteResult(workflow_name=None, response="")
