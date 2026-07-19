"""Arm C — prompt evolution: distill the day's conversations into a persistent
learned-preferences block appended to the system prompt (eval path only).

TextGrad-style single nightly call: a strong model reads the current block plus
the day's exchanges (with any feedback) and rewrites the block. The block is
size-capped so it can't grow into a context hog, and versioned like every
other artifact. Production's prompt is untouched until an arm wins the
pre-registered bar.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from research import artifacts
from research.db import ResearchStore

logger = logging.getLogger(__name__)

ARM = "prompt"
MAX_BLOCK_CHARS = 2500
EVOLVER_MODEL = "claude-sonnet-5"

_EVOLVE_PROMPT = """You maintain the LEARNED PREFERENCES block of a butler-style personal assistant's system prompt. It records durable, useful facts about this one user learned from conversations: preferences, routines, corrections, phrasing they respond well to.

Current block (may be empty):
<block>
{block}
</block>

Today's conversations (user/assistant pairs; feedback lines mark exchanges the user explicitly or implicitly liked (+1) or disliked (-1)):
<conversations>
{conversations}
</conversations>

Rewrite the block:
- Add durable facts evidenced today; strengthen or correct existing entries; drop entries contradicted today. Disliked exchanges are evidence about what to avoid.
- Never invent facts without evidence in the conversations. Never store secrets, credentials, addresses, or card details.
- Terse bullet lines, most useful first. HARD LIMIT {max_chars} characters — consolidate rather than truncate.

Answer with only a JSON object:
{{"block": "<the full revised block>", "changelog": "<one line describing what changed and why>"}}"""


def format_conversations(store: ResearchStore, since_ts: float, *,
                         max_exchanges: int = 60) -> str:
    """Compact transcript of the day's chat exchanges with feedback markers."""
    lines = []
    for e in store.exchanges_since(since_ts)[-max_exchanges:]:
        if e["route"] != "chat":
            continue
        lines.append(f"USER: {e['user_text']}")
        lines.append(f"ASSISTANT: {e['reply_text']}")
        for fb in store.feedback_for(e["id"]):
            lines.append(f"FEEDBACK: {fb['signal']:+d} ({fb['source']})")
        lines.append("")
    return "\n".join(lines).strip()


def _default_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=EVOLVER_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def load_current_block(artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR) -> str:
    path = artifacts.current_path(ARM, artifacts_dir)
    if path is None:
        return ""
    block_file = path / "block.md"
    return block_file.read_text() if block_file.exists() else ""


def evolve(
    store: ResearchStore,
    since_ts: float,
    date_str: str,
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
    artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR,
) -> Optional[str]:
    """Run one evolution step. Returns the new version name, or None if skipped."""
    conversations = format_conversations(store, since_ts)
    if not conversations:
        logger.info("prompt evolver: no chat exchanges since cutoff — skipping")
        return None

    current = load_current_block(artifacts_dir)
    prompt = _EVOLVE_PROMPT.format(block=current or "(empty)",
                                   conversations=conversations,
                                   max_chars=MAX_BLOCK_CHARS)
    raw = (llm_fn or _default_llm)(prompt)

    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        block = str(obj["block"]).strip()
        changelog = str(obj.get("changelog", "")).strip()
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        logger.warning("prompt evolver: unparseable response (%s) — keeping current block", e)
        return None

    if len(block) > MAX_BLOCK_CHARS:
        logger.warning("prompt evolver: block over cap (%d chars) — hard-truncating",
                       len(block))
        block = block[:MAX_BLOCK_CHARS]

    version_dir = artifacts.new_version(ARM, date_str, artifacts_dir)
    (version_dir / "block.md").write_text(block)
    (version_dir / "changelog.txt").write_text(changelog + "\n")
    diff = "\n".join(difflib.unified_diff(
        current.splitlines(), block.splitlines(),
        fromfile="previous", tofile=version_dir.name, lineterm="",
    ))
    (version_dir / "diff.patch").write_text(diff + "\n")
    artifacts.advance_current(ARM, version_dir.name, artifacts_dir)
    logger.info("prompt evolver: %s (%d chars) — %s", version_dir.name, len(block), changelog)
    return version_dir.name


def system_block(artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR) -> str:
    """The block as injected into the eval/replay system prompt ('' when none)."""
    block = load_current_block(artifacts_dir).strip()
    return f"LEARNED PREFERENCES:\n{block}" if block else ""
