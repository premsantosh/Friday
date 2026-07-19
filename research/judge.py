"""Judges for pairwise response comparison.

A judge sees one eval prompt, its annotations (known preferences, style
constraints), and two candidate responses labeled A/B, and picks a winner.
Position bias is handled by the eval runner (each pair is judged in both
orders); judges themselves are stateless single-shot calls.

SonnetJudge is the primary (paid API); QwenJudge re-judges a sample locally to
report agreement; FakeJudge keeps tests hermetic.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

JUDGE_MODEL_SONNET = "claude-sonnet-5"
JUDGE_MODEL_QWEN = "qwen3:8b"

_RUBRIC = """You are judging which of two assistant responses better serves this specific user.

The assistant is Jarvis: a butler-style personal assistant. House style: addresses the user as "sir", composed deadpan delivery, no exclamation marks, at most ~3 sentences for simple requests, British vocabulary.

Known facts about this user relevant to the prompt (may be empty):
{annotations}

User prompt:
{prompt}

Response A:
{a}

Response B:
{b}

Judge which response is better for THIS user: correctness on the user's known preferences first, then house style, then helpfulness. A response that ignores a known preference loses to one that honors it. Do not reward length.

Answer with only a JSON object: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


@dataclass
class Verdict:
    winner: str  # "A" | "B" | "tie" | "error"
    reason: str = ""
    judge: str = ""


class Judge(Protocol):
    name: str

    def judge(self, prompt: str, annotations: str, a: str, b: str) -> Verdict: ...


def _parse_verdict(text: str, judge_name: str) -> Verdict:
    """Extract the verdict JSON from a judge's reply (tolerates surrounding prose)."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        winner = str(obj.get("winner", "")).strip().upper()
        if winner not in ("A", "B", "TIE"):
            return Verdict("error", f"bad winner {winner!r}", judge_name)
        return Verdict("tie" if winner == "TIE" else winner,
                       str(obj.get("reason", ""))[:300], judge_name)
    except (ValueError, json.JSONDecodeError):
        return Verdict("error", f"unparseable: {text[:80]!r}", judge_name)


class SonnetJudge:
    """Primary judge (paid API call per comparison)."""

    def __init__(self, model: str = JUDGE_MODEL_SONNET):
        import anthropic
        self.name = f"sonnet:{model}"
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def judge(self, prompt: str, annotations: str, a: str, b: str) -> Verdict:
        content = _RUBRIC.format(annotations=annotations or "(none)", prompt=prompt, a=a, b=b)
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": content}],
            )
            return _parse_verdict(resp.content[0].text, self.name)
        except Exception as e:
            logger.warning("Sonnet judge failed: %s", e)
            return Verdict("error", str(e)[:120], self.name)


class QwenJudge:
    """Local agreement auditor via Ollama (free, weaker)."""

    def __init__(self, model: str = JUDGE_MODEL_QWEN,
                 base_url: str = "http://localhost:11434", poster=None):
        self.name = f"qwen:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._poster = poster

    def judge(self, prompt: str, annotations: str, a: str, b: str) -> Verdict:
        content = _RUBRIC.format(annotations=annotations or "(none)", prompt=prompt, a=a, b=b)
        poster = self._poster
        if poster is None:
            import requests
            poster = requests.post
        try:
            resp = poster(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "stream": False, "think": False,
                      "messages": [{"role": "user", "content": content}]},
                timeout=120,
            )
            text = (resp.json().get("message") or {}).get("content", "")
            return _parse_verdict(text, self.name)
        except Exception as e:
            logger.warning("Qwen judge failed: %s", e)
            return Verdict("error", str(e)[:120], self.name)


class FakeJudge:
    """Deterministic judge for tests and --dry-run: prefers the longer response,
    or an explicit scripted sequence."""

    def __init__(self, script: Optional[list[str]] = None):
        self.name = "fake"
        self._script = list(script) if script else None
        self.calls: list[tuple[str, str, str]] = []

    def judge(self, prompt: str, annotations: str, a: str, b: str) -> Verdict:
        self.calls.append((prompt, a, b))
        if self._script:
            return Verdict(self._script.pop(0), "scripted", self.name)
        if len(a) == len(b):
            return Verdict("tie", "equal length", self.name)
        return Verdict("A" if len(a) > len(b) else "B", "longer", self.name)
