"""Proactive surfacing of self-diagnosed problems.

The nightly launchd job discards `python -m research nightly`'s exit code, so
a run that fails every stage used to be invisible until someone remembered to
run `research status`. `cmd_nightly` now calls `format_nightly_alert` +
`send_telegram` after a failed run — a silent no-op when Telegram isn't
configured, and never an exception (an alerting failure must not make the
nightly itself look broken).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def format_nightly_alert(stage_status: Dict[str, str]) -> Optional[str]:
    """A short failure message naming the failed stages, or None if all ok.

    Stage notes are code-generated ("FAILED: ValueError: ...") — no user text.
    """
    failed = {stage: note for stage, note in stage_status.items()
              if str(note).startswith("FAILED")}
    if not failed:
        return None
    lines = ["Friday nightly run had failures:"]
    for stage, note in failed.items():
        lines.append(f"  {stage}: {note}")
    lines.append("Run `python -m research doctor` for a full self-diagnosis.")
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    """Best-effort Telegram notification to the owner. Never raises."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_NOTIFY_CHAT_ID")
    if not chat_id:
        allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        chat_id = allowed.split(",")[0].strip() if allowed.strip() else ""
    if not token or not chat_id:
        return False
    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        logger.warning("Nightly alert delivery failed", exc_info=True)
        return False
