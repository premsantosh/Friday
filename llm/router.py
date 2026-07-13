"""
Intent router: uses the configured LLM to classify user input into a workflow
and extract entities. Classification only — the router never drafts spoken
replies. Conversational turns are answered by the personality LLM so the
assistant's voice is consistent on every path.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an intent classifier for a smart home voice assistant.
Determine if the user's request maps to one of the available workflows and
extract relevant entities.

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "workflow": "<workflow_name or null>",
  "entities": {}
}

Entities capture the specifics of the command, e.g. for lights:
{"action": "on"|"off"|"dim", "room": "bedroom", "mood": "relax"}.
Set "workflow" to null when no workflow fits — the assistant will answer
conversationally instead."""

_EXTRACT_PROMPT = """You are an entity extractor for a smart home voice assistant.
You are given ONE candidate workflow and a user utterance. Decide whether the
utterance really is a command for that workflow, and if so extract its entities
(e.g. {"action": "off", "room": "bedroom"}). Extract only what the utterance
actually says — never invent defaults.

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "matches": true or false,
  "entities": {}
}"""


@dataclass
class RouteResult:
    workflow_name: Optional[str]
    entities: dict = field(default_factory=dict)


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
            raw = self._call_llm(_SYSTEM_PROMPT, user_message)
            return self._parse(raw)
        except Exception:
            logger.exception("Intent router failed for: %s", text)
            return RouteResult(workflow_name=None)

    def extract_entities(self, text: str, workflow) -> Optional[dict]:
        """Verify a keyword-matched workflow against the utterance and extract
        its entities.

        Returns {"matches": bool, "entities": dict}, or None when the LLM call
        or parse fails (caller should fall back to full routing rather than
        execute with empty entities).
        """
        examples = ", ".join(f'"{ex}"' for ex in workflow.trigger.examples[:3])
        user_message = (
            f"Candidate workflow: {workflow.name}\n"
            f"Description: {workflow.description}\n"
            f"Example commands: {examples}\n\n"
            f"User said: \"{text}\""
        )
        try:
            raw = self._call_llm(_EXTRACT_PROMPT, user_message)
            data = json.loads(self._strip_to_json(raw))
            return {
                "matches": bool(data.get("matches")),
                "entities": data.get("entities") or {},
            }
        except Exception:
            logger.exception("Entity extraction failed for workflow %s: %s",
                             workflow.name, text)
            return None

    def _build_workflow_context(self, workflow_manager) -> str:
        lines = ["Available workflows:"]
        for workflow in workflow_manager.workflows.values():
            examples = ", ".join(f'"{ex}"' for ex in workflow.trigger.examples[:3])
            lines.append(f"  {workflow.name}: {workflow.description}. Examples: {examples}")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        provider = self.config.provider
        if provider == "anthropic":
            return self._call_anthropic(system_prompt, user_message)
        if provider == "openai":
            return self._call_openai(system_prompt, user_message)
        if provider == "ollama":
            return self._call_ollama(system_prompt, user_message)
        raise ValueError(f"Unsupported LLM provider for routing: {provider}")

    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        if self._anthropic_client is None:
            import anthropic
            api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)

        response = self._anthropic_client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        if self._openai_client is None:
            from openai import OpenAI
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = OpenAI(api_key=api_key)

        response = self._openai_client.chat.completions.create(
            model=self.config.openai_model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        import requests

        response = requests.post(
            f"{self.config.ollama_base_url}/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 300},
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @staticmethod
    def _strip_to_json(raw: str) -> str:
        text = (raw or "").strip()
        # Strip the markdown code fences Claude sometimes adds (```json ... ```).
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text).strip()
        # Be liberal about surrounding prose: keep the outermost {...} object.
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                text = text[start:end + 1]
        return text

    def _parse(self, raw: str) -> RouteResult:
        try:
            data = json.loads(self._strip_to_json(raw))
            return RouteResult(
                workflow_name=data.get("workflow") or None,
                entities=data.get("entities") or {},
            )
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Router returned unparseable response: %s", raw[:200])
            return RouteResult(workflow_name=None)
