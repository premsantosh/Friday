"""Commit the nightly evidence (aggregates only) to git, behind a PII gate.

results/eval.csv and results/nightly/*.md are the study's longitudinal record;
leaving them uncommitted means the evidence lives only in the working tree.
The repo is public, so every commit is gated by a pattern scan first: emails,
phone-like numbers, and a user-configured name list (FRIDAY_PII_NAMES env or
~/.friday/pii_names.txt — names are never hardcoded here). Findings name the
file, line and kind only, never the matched text.

Commit only, never push.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.\w[\w.-]+")
# Phone-like: 9+ digit runs with separators. Excludes the numbers results are
# made of: dates (20260830, 2026-08-30), decimals (0.5000), and plain integer
# ids — those never carry separators in phone shapes or a leading +.
_PHONE_RE = re.compile(
    r"(?<![\d.])(?:\+\d{10,15}|\d{3}[\s().-]\d{3}[\s().-]?\d{4,})(?![\d.])")
_NAMES_FILE = Path("~/.friday/pii_names.txt").expanduser()

RESULT_PATHS = ("eval.csv", "nightly")


def guarded_names() -> list[str]:
    raw = os.getenv("FRIDAY_PII_NAMES", "")
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if not names and _NAMES_FILE.exists():
        try:
            names = [line.strip() for line in _NAMES_FILE.read_text().splitlines()
                     if line.strip() and not line.startswith("#")]
        except OSError:
            pass
    return names


def pii_findings(paths: list[Path], names: list[str] | None = None) -> list[str]:
    """['file:line kind', ...] — never includes the matched text itself."""
    names = guarded_names() if names is None else names
    name_res = [re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE) for n in names]
    findings: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if _EMAIL_RE.search(line):
                findings.append(f"{path.name}:{i} email")
            if _PHONE_RE.search(line):
                findings.append(f"{path.name}:{i} phone-like")
            if any(r.search(line) for r in name_res):
                findings.append(f"{path.name}:{i} guarded-name")
    return findings


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                          text=True, timeout=30)


def commit_results(results_dir: Path, date_str: str) -> str:
    """Commit results/eval.csv + results/nightly/ (pathspec-only, never push)."""
    top = _git(results_dir, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return "skipped: not a git repo"
    repo_root = Path(top.stdout.strip())
    specs = [str(results_dir / p) for p in RESULT_PATHS
             if (results_dir / p).exists()]
    if not specs:
        return "skipped: no result files"
    status = _git(repo_root, "status", "--porcelain", "--", *specs)
    if status.returncode != 0:
        return f"skipped: git status failed ({status.stderr.strip()[:80]})"
    if not status.stdout.strip():
        return "nothing to commit"
    add = _git(repo_root, "add", "--", *specs)
    if add.returncode != 0:
        return f"skipped: git add failed ({add.stderr.strip()[:80]})"
    commit = _git(repo_root, "commit", "-m", f"results: nightly {date_str}",
                  "--", *specs)
    if commit.returncode != 0:
        return f"skipped: git commit failed ({commit.stderr.strip()[:80]})"
    rev = _git(repo_root, "rev-parse", "--short", "HEAD").stdout.strip()
    return f"committed {rev}"


def scan_targets(results_dir: Path) -> list[Path]:
    targets = []
    csv_path = results_dir / "eval.csv"
    if csv_path.exists():
        targets.append(csv_path)
    nightly = results_dir / "nightly"
    if nightly.is_dir():
        targets.extend(sorted(nightly.glob("*.md")))
    return targets
