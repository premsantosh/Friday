"""Versioned artifact directories shared by all arms.

Layout: <artifacts_dir>/<arm>/vYYYYMMDD[-N]/... plus a `current` pointer file
(plain text, atomically replaced) naming the active version. Old versions are
immutable; revert = rewrite the pointer. A pointer file beats a symlink here:
inspectable with cat, diffable, and os.replace is atomic on APFS.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_ARTIFACTS_DIR = Path("~/.friday/research").expanduser()


def arm_dir(arm: str, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir).expanduser() / arm


def new_version(arm: str, date_str: str,
                artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Create and return a fresh immutable version dir (vYYYYMMDD, -2, -3... on rerun)."""
    base = arm_dir(arm, artifacts_dir)
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / f"v{date_str}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = base / f"v{date_str}-{n}"
    candidate.mkdir()
    return candidate


def current_version(arm: str,
                    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> str | None:
    pointer = arm_dir(arm, artifacts_dir) / "current"
    if not pointer.exists():
        return None
    name = pointer.read_text().strip()
    return name or None


def current_path(arm: str,
                 artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path | None:
    name = current_version(arm, artifacts_dir)
    if name is None:
        return None
    path = arm_dir(arm, artifacts_dir) / name
    return path if path.exists() else None


def advance_current(arm: str, version_name: str,
                    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
    """Atomically point `current` at version_name (which must exist)."""
    base = arm_dir(arm, artifacts_dir)
    if not (base / version_name).exists():
        raise FileNotFoundError(f"{arm}/{version_name} does not exist")
    fd, tmp = tempfile.mkstemp(dir=base, prefix=".current-")
    with os.fdopen(fd, "w") as f:
        f.write(version_name + "\n")
    os.replace(tmp, base / "current")


def list_versions(arm: str,
                  artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> list[str]:
    base = arm_dir(arm, artifacts_dir)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("v"))
