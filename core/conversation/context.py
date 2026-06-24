"""
Layer A — Conversational Context Register.

A thin adapter over the `turnstile-ctx` ContextRegister. It carries the last
route's domain / action / parameters forward for a few turns so follow-up
utterances ("turn it down", "make it 7:30") can be resolved against the prior
turn by the LLM router.

Everything turnstile-specific is isolated here so the rest of Friday never
imports it directly, and so a missing or broken dependency degrades to a no-op
instead of breaking the assistant. By design, enrich()/update() never raise.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationContext:
    """Short-term routing context. No-op if turnstile-ctx is unavailable."""

    def __init__(self, persist_path: Optional[str] = None, max_turns: int = 3):
        self._register = None
        self._routing_result_cls = None

        try:
            # turnstile-ctx installs as the top-level `src` package.
            from src import ContextRegister, RegisterConfig, RoutingResult

            config = RegisterConfig(
                max_turns=max_turns,
                enable_duckling=False,  # avoid the external Duckling server dependency
                enable_persistence=persist_path is not None,
                persistence_path=persist_path,
            )
            self._register = ContextRegister(config)
            self._routing_result_cls = RoutingResult
            logger.info("ConversationContext enabled (persist=%s)", bool(persist_path))
        except Exception:
            logger.warning(
                "turnstile-ctx unavailable; conversational context disabled", exc_info=True
            )

    @property
    def enabled(self) -> bool:
        return self._register is not None

    def enrich(self, text: str) -> str:
        """Prefix `text` with carried context for routing. Returns `text` unchanged on any issue."""
        if self._register is None:
            return text
        try:
            return self._register.enrich(text).enriched_utterance
        except Exception:
            logger.debug("enrich failed; using raw text", exc_info=True)
            return text

    def update(self, workflow_name: Optional[str], entities: Optional[dict] = None,
               text: str = "") -> None:
        """Record a successful route so the next turn inherits its domain/params."""
        if self._register is None or self._routing_result_cls is None or not workflow_name:
            return
        try:
            entities = entities or {}
            result = self._routing_result_cls(
                action_name=workflow_name,
                domain=workflow_name,
                device=entities.get("device"),
                parameters=entities or None,
            )
            self._register.update(result, text)
        except Exception:
            logger.debug("context update failed", exc_info=True)

    def clear(self) -> None:
        if self._register is None:
            return
        try:
            self._register.clear()
        except Exception:
            logger.debug("context clear failed", exc_info=True)
