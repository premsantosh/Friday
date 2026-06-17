"""
Structured LLM tasks — schema + grounding + fallback, declaratively
(harness spec §8).

An `LLMTask` declares what a model may contribute: the output fields, their
deterministic validators, which of them must be *grounded* (the value must
literally appear in the supplied evidence — the anti-hallucination rule), and
the egress sink the payload is checked against before anything is sent.

`run_task` makes the one-shot JSON call and validates every field; an invalid
or ungrounded value is dropped, and a task that fails its required fields is
indistinguishable from no LLM at all — the caller's deterministic fallback
always decides. Results carry provenance ("llm" / "fallback" / "none").
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .egress import Sink, guard

logger = logging.getLogger(__name__)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Tolerates code fences and prose around the JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class JsonLLMClient:
    """One-shot, history-free JSON completions against the Anthropic API.
    Returns None on any failure — callers always have a deterministic path."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls, *, model_env: str, default_model: str,
                 api_key_env: str = "ANTHROPIC_API_KEY") -> Optional["JsonLLMClient"]:
        api_key = os.getenv(api_key_env)
        if not api_key:
            return None
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic SDK unavailable; LLM tasks disabled")
            return None
        return cls(anthropic.Anthropic(api_key=api_key),
                   os.getenv(model_env, default_model))

    def complete_json(self, system: str, user: str,
                      max_tokens: int = 700) -> Optional[Dict[str, Any]]:
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            return parse_json_object(text)
        except Exception:
            logger.warning("LLM task call failed", exc_info=True)
            return None


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: type = str
    max_len: int = 300
    required: bool = False          # invalid/missing required field fails the task
    grounded: bool = False          # value must appear verbatim in the evidence
    valid: Optional[Callable[[Any], bool]] = None
    reject_if: Optional[Callable[[Any], bool]] = None
    coerce: Optional[Callable[[Any], Any]] = None   # runs before validation


@dataclass(frozen=True)
class LLMTask:
    name: str
    system_prompt: str
    fields: Tuple[FieldSpec, ...]
    sink: Sink
    max_tokens: int = 300
    # Cross-field validation/defaults; return None to reject the whole result.
    finalize: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None


@dataclass
class TaskResult:
    values: Optional[Dict[str, Any]]
    provenance: str                 # "llm" | "fallback" | "none"

    @property
    def from_llm(self) -> bool:
        return self.provenance == "llm"


def _validate_field(spec: FieldSpec, raw: Any, evidence: Optional[str]) -> Optional[Any]:
    """The validated value, or None when the proposal fails any check."""
    if raw is None:
        return None
    value = raw
    if spec.coerce is not None:
        try:
            value = spec.coerce(value)
        except (TypeError, ValueError):
            return None
        if value is None:
            return None
    if spec.type is str:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or len(value) > spec.max_len:
            return None
        if value.lower() in ("null", "none", "n/a"):
            return None
    elif not isinstance(value, spec.type):
        return None
    if spec.grounded and (evidence is None or str(value) not in evidence):
        return None     # the model named something the evidence doesn't contain
    if spec.valid is not None and not spec.valid(value):
        return None
    if spec.reject_if is not None and spec.reject_if(value):
        return None
    return value


def run_task(client, task: LLMTask, payload: Any,
             evidence: Optional[str] = None,
             fallback: Optional[Callable[[], Optional[Dict[str, Any]]]] = None) -> TaskResult:
    """Execute the task. The egress sink is checked before the call; every
    output field is validated deterministically; failure → fallback/none."""

    def _fall() -> TaskResult:
        if fallback is not None:
            values = fallback()
            if values is not None:
                return TaskResult(values, "fallback")
        return TaskResult(None, "none")

    if client is None:
        return _fall()

    guard(task.sink, payload)
    user = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    raw = client.complete_json(task.system_prompt, user, max_tokens=task.max_tokens)
    if raw is None:
        return _fall()

    values: Dict[str, Any] = {}
    for spec in task.fields:
        value = _validate_field(spec, raw.get(spec.name), evidence)
        if value is None:
            if spec.required:
                return _fall()
            continue
        values[spec.name] = value

    if task.finalize is not None:
        values = task.finalize(values)
        if values is None:
            return _fall()
    if not values:
        return _fall()
    return TaskResult(values, "llm")
