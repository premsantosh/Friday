"""
Optional LangSmith tracing for the agent graph (free tier friendly).

Off unless `AgentConfig.tracing` (FRIDAY_LANGSMITH_TRACING=true) *and*
LANGSMITH_API_KEY are both set — Friday's own flag, not the library-global
LANGSMITH_TRACING, so nothing traces implicitly. The tracer is passed per
invocation via `config={"callbacks": [...]}` on the agent graph only; legacy
paths and the reservations workflow never emit traces.

Privacy, two layers, applied in `hide_inputs`/`hide_outputs` before upload:
  1. every string passes through the harness `redact_text` (cards, e-mails,
     phone numbers — Friday's log-redaction posture extends to traces);
  2. the memory context block (long-term facts, search context) is hidden
     entirely: the `context_block` state field and the
     <friday_context>…</friday_context> span of the system prompt are replaced
     by a placeholder. Tool calls/args/results and the user text stay visible
     (redacted) — that is what debugging routing needs.

Free tier (checked 2026-08-22): 5k traces/month, 14-day retention; with no card
on file overage returns 429 and traces are dropped, never billed. Throttle with
LANGSMITH_TRACING_SAMPLING_RATE if usage ever approaches the cap.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

from config import AgentConfig
from core.harness import redact_text

logger = logging.getLogger(__name__)

CONTEXT_OPEN = "<friday_context>"
CONTEXT_CLOSE = "</friday_context>"
_CONTEXT_SPAN_RE = re.compile(re.escape(CONTEXT_OPEN) + r".*?" + re.escape(CONTEXT_CLOSE), re.DOTALL)
_HIDDEN_KEYS = {"context_block"}


def _placeholder(n: int) -> str:
    return f"<context_block hidden, {n} chars>"


def hide_context_spans(text: str) -> str:
    return _CONTEXT_SPAN_RE.sub(lambda m: _placeholder(len(m.group(0))), text)


def scrub(obj: Any, _depth: int = 0) -> Any:
    """Recursively redact PII and hide the context block in a trace payload.
    Returns a new structure; never mutates the original (LangSmith passes us
    the live run dict)."""
    if _depth > 30:
        return "<truncated>"
    if isinstance(obj, str):
        return redact_text(hide_context_spans(obj))
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _HIDDEN_KEYS and isinstance(v, str):
                out[k] = _placeholder(len(v))
            else:
                out[k] = scrub(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub(v, _depth + 1) for v in obj]
    # Message objects etc. — LangSmith serialises them before calling the hide
    # hooks, so we normally see plain dicts. Anything else passes through.
    return obj


def hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def build_tracer(config: AgentConfig):
    """Return a LangChainTracer, or None when tracing is off / unconfigured."""
    if not config.tracing:
        return None
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning("FRIDAY_LANGSMITH_TRACING is on but LANGSMITH_API_KEY is unset; tracing disabled")
        return None
    try:
        from langchain_core.tracers import LangChainTracer
        from langsmith import Client
    except Exception:  # pragma: no cover - langsmith is a transitive dep of langchain-core
        logger.warning("langsmith unavailable; tracing disabled", exc_info=True)
        return None

    client = Client(
        api_key=api_key,
        hide_inputs=scrub,
        hide_outputs=scrub,
        tracing_sampling_rate=config.tracing_sampling_rate,
    )
    return LangChainTracer(project_name=config.tracing_project, client=client)


def trace_config(tracer, *, user_id: str, thread_id: str, run_kind: str) -> dict:
    """Per-invocation RunnableConfig additions. Returns {} when tracing is off
    so the engine can merge it unconditionally."""
    if tracer is None:
        return {}
    return {
        "callbacks": [tracer],
        "metadata": {
            "user_id_hash": hash_user_id(user_id),
            "thread_id": thread_id,
            "engine": "langgraph",
            "run_kind": run_kind,
        },
        "tags": [f"run:{run_kind}"],
    }
