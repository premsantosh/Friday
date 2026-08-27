"""Maintenance operations shared by the research CLI and Friday's self-repair.

`python -m research revert` and the `self_repair` workflow must do exactly the
same thing — one implementation, one `artifact.advanced` event — so the
provenance trail cannot diverge depending on who asked for the revert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def revert_arm(arm: str, to: str, *, artifacts_dir: Optional[Path] = None,
               db_path: str = "~/.friday/research.db",
               via: str = "manual revert") -> Optional[str]:
    """Repoint an arm's `current` at an existing version and record the event.

    Returns the previous version name. Raises ValueError for an unknown
    version — callers phrase their own refusal.
    """
    from research import artifacts
    from research.db import ResearchStore

    art_dir = (Path(artifacts_dir).expanduser() if artifacts_dir
               else artifacts.DEFAULT_ARTIFACTS_DIR)
    versions = artifacts.list_versions(arm, art_dir)
    if to not in versions:
        raise ValueError(
            f"unknown version {to!r} for arm {arm!r}; "
            f"available: {', '.join(versions) or 'none'}")
    previous = artifacts.current_version(arm, art_dir)
    artifacts.advance_current(arm, to, art_dir)
    store = ResearchStore(db_path)
    try:
        store.emit("artifact.advanced", subject_type="artifact",
                   subject_id=f"{arm}/{to}", arm=arm,
                   artifact_version=f"{arm}/{to}",
                   detail={"previous": previous, "via": via})
    finally:
        store.close()
    return previous
