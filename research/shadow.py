"""ShadowRunner: local model silently answers the same chat messages.

The production reply (Haiku) has already been sent by the time a job lands
here; this only generates the local model's answer for later evaluation. It
must therefore never block or surface an error into the user-facing path:
enqueue drops on overflow, the worker swallows every exception at debug level.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

from research.db import ResearchStore
from research.generate import strip_think

logger = logging.getLogger(__name__)

_STOP = object()


def supports_thinking(model_tag: str) -> bool:
    """Whether the Ollama model accepts the `think` option (passing it to a
    non-thinking model is an API error, so it must be conditional)."""
    return any(family in model_tag.lower() for family in ("qwen3", "deepseek-r1"))


class ShadowRunner:
    def __init__(
        self,
        store: ResearchStore,
        model_tag: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        poster: Optional[Callable] = None,
        timeout_s: float = 60.0,
        max_queue: int = 16,
    ):
        self.store = store
        self.model_tag = model_tag
        self.base_url = base_url.rstrip("/")
        self._poster = poster  # callable(url, json=..., timeout=...) -> resp
        self.timeout_s = timeout_s
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="shadow-runner")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass  # daemon thread; process exit will take it down
        self._thread.join(timeout=5)
        self._thread = None

    def enqueue(self, exchange_id: int) -> bool:
        """Queue an exchange for shadow generation. Drops when full."""
        try:
            self._queue.put_nowait(exchange_id)
            return True
        except queue.Full:
            logger.debug("Shadow queue full — dropping exchange %s", exchange_id)
            return False

    # ------------------------------------------------------------- internals
    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP:
                return
            try:
                self._process(job)
            except Exception:
                logger.debug("Shadow generation failed for exchange %s", job, exc_info=True)

    def _process(self, exchange_id: int) -> None:
        exchange = self.store.get_exchange(exchange_id)
        if exchange is None:
            return
        snapshot = exchange.get("context_snapshot")
        if not snapshot:
            return
        messages = [{"role": "system", "content": snapshot.get("system_prompt", "")}]
        messages.extend(snapshot.get("messages", []))

        poster = self._poster
        if poster is None:
            import requests
            poster = requests.post

        payload = {"model": self.model_tag, "messages": messages, "stream": False}
        if supports_thinking(self.model_tag):
            payload["think"] = False  # answer directly; reasoning isn't the response

        t0 = time.monotonic()
        resp = poster(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        status = int(getattr(resp, "status_code", 200))
        if not (200 <= status < 300):
            logger.debug("Shadow Ollama call returned HTTP %s", status)
            return
        text = strip_think((resp.json().get("message") or {}).get("content", ""))
        if not text:
            return
        self.store.add_shadow_response(
            exchange_id,
            arm="base",
            mode="live",
            response_text=text,
            model_tag=self.model_tag,
            gen_ms=int((time.monotonic() - t0) * 1000),
        )
