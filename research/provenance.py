"""Per-artifact provenance manifests: exactly what a version consumed.

The events table is a timeline (when things happened, in what order). This is
the other half: an immutable record sitting next to the artifact itself saying
which exchanges, feedback rows and corrections went into it. Written once at
build time and never updated, so `revert --to v20260726` also reverts the
answer to "what was this trained on".

Both exist on purpose. Reconstructing membership by scanning events would be
fragile and expensive, and emitting a per-exchange event nightly for a corpus
that is retrained from scratch each night would re-emit the whole history
forever (see research/events.py on the state-transition rule).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from research import artifacts

logger = logging.getLogger(__name__)

MANIFEST_NAME = "provenance.json"


def write_manifest(version_dir: Path, arm: str, **fields: Any) -> Path:
    """Write <version_dir>/provenance.json. Immutable once written."""
    path = Path(version_dir) / MANIFEST_NAME
    payload = {"artifact": f"{arm}/{Path(version_dir).name}", "arm": arm, **fields}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def read_manifest(
    arm: str,
    version: str,
    artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR,
) -> Optional[dict]:
    """The manifest for one version, or None when it predates provenance."""
    path = artifacts.arm_dir(arm, artifacts_dir) / version / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Unreadable provenance manifest: %s", path, exc_info=True)
        return None


def git_rev() -> str:
    """Short HEAD sha, or '' outside a repo. Pins the code that built an artifact."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""
