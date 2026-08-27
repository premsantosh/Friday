"""Provider registry behind Friday's self-awareness.

Two aggregators — `gather_status()` (what is my state?) and `run_doctor()`
(is anything wrong?) — iterate registered `StatusProvider`s rather than a
hardcoded list, so a subsystem added later is covered by adding one provider
next to its own code (or, for workflows, by implementing `status_snapshot()` /
`health_checks()` on the workflow class; see workflows/base.py).

Invariants every provider must keep:
  - Read-only. Never create files or databases — this code runs in ephemeral
    modes (--chat/--test) where persistence must not spring into existence.
    Open SQLite via `file:...?mode=ro` URIs and return {"available": False}
    when a store is absent.
  - Text-free. Snapshots and check messages may carry counts, ids, stage
    notes, event names, sizes and versions — never user text, reply text, or
    memory facts (same rule the research events table enforces).
  - Never raise. The aggregators wrap each provider anyway (a broken provider
    becomes a FAIL check, not a crash), but individual checks should degrade
    to WARN/SKIP on their own where they can.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class CheckStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str


@dataclass
class Paths:
    """Where Friday's state lives. Override wholesale in tests."""

    state_dir: Path          # ~/.friday — globbed for *.db and stores
    research_db: Path
    artifacts_dir: Path      # ~/.friday/research — arms, logs, lock, backups
    audit_db: Path

    @classmethod
    def default(cls) -> "Paths":
        # Late imports: config modules, not heavy ones, but keep import time low.
        from core.harness.audit import DEFAULT_PATH as AUDIT_DEFAULT
        from research.artifacts import DEFAULT_ARTIFACTS_DIR

        state_dir = Path("~/.friday").expanduser()
        return cls(
            state_dir=state_dir,
            research_db=state_dir / "research.db",
            artifacts_dir=Path(DEFAULT_ARTIFACTS_DIR).expanduser(),
            audit_db=Path(AUDIT_DEFAULT).expanduser(),
        )


def _default_launchctl(args: List[str]):
    """Run launchctl if present; None means 'not this platform'."""
    if shutil.which("launchctl") is None:
        return None
    return subprocess.run(["launchctl", *args], capture_output=True, text=True,
                          timeout=10)


def _default_http_get(url: str, timeout: float) -> int:
    import requests

    return requests.get(url, timeout=timeout).status_code


@dataclass
class Probes:
    """Injectable environment probes so providers are testable offline."""

    launchctl: Callable[[List[str]], Any] = _default_launchctl
    http_get: Callable[[str, float], int] = _default_http_get
    now: Callable[[], float] = time.time


class StatusProvider:
    """One subsystem's view of itself. Subclass, set `name`, override either hook."""

    name: str = "provider"

    def snapshot(self, paths: Paths, probes: Probes) -> Dict[str, Any]:
        return {}

    def checks(self, paths: Paths, probes: Probes) -> List[CheckResult]:
        return []


_PROVIDERS: List[StatusProvider] = []


def register_provider(provider: StatusProvider) -> StatusProvider:
    """Add a provider (replacing any earlier one with the same name)."""
    _PROVIDERS[:] = [p for p in _PROVIDERS if p.name != provider.name]
    _PROVIDERS.append(provider)
    return provider


def iter_providers() -> List[StatusProvider]:
    _ensure_core_providers()
    return list(_PROVIDERS)


def _ensure_core_providers() -> None:
    # Importing the module self-registers the core providers exactly once.
    import introspection.providers  # noqa: F401


# ------------------------------------------------------------------ aggregators

def gather_status(paths: Optional[Paths] = None, probes: Optional[Probes] = None,
                  workflow_manager=None) -> Dict[str, Any]:
    """Every provider's snapshot, keyed by provider name. Never raises."""
    paths = paths or Paths.default()
    probes = probes or Probes()
    out: Dict[str, Any] = {}
    for provider in iter_providers():
        try:
            out[provider.name] = provider.snapshot(paths, probes)
        except Exception as exc:  # a broken provider must not hide the rest
            out[provider.name] = {"available": False,
                                  "error": type(exc).__name__}
    if workflow_manager is not None:
        out["workflows"] = _workflow_snapshots(workflow_manager)
    return out


def run_doctor(paths: Optional[Paths] = None, probes: Optional[Probes] = None,
               workflow_manager=None) -> List[CheckResult]:
    """Every provider's checks plus every registered workflow's own
    health_checks(). A provider or workflow that raises becomes a FAIL row."""
    paths = paths or Paths.default()
    probes = probes or Probes()
    results: List[CheckResult] = []
    for provider in iter_providers():
        try:
            results.extend(provider.checks(paths, probes))
        except Exception as exc:
            results.append(CheckResult(f"{provider.name}.provider",
                                       CheckStatus.FAIL,
                                       f"provider crashed: {type(exc).__name__}: {exc}"))
    if workflow_manager is not None:
        for name, wf in getattr(workflow_manager, "workflows", {}).items():
            try:
                results.extend(wf.health_checks() or [])
            except Exception as exc:
                results.append(CheckResult(f"workflow.{name}", CheckStatus.FAIL,
                                           f"health_checks crashed: {type(exc).__name__}: {exc}"))
    return results


def _workflow_snapshots(workflow_manager) -> Dict[str, Any]:
    snapshots: Dict[str, Any] = {}
    for name, wf in getattr(workflow_manager, "workflows", {}).items():
        try:
            snap = wf.status_snapshot()
        except Exception as exc:
            snap = {"available": False, "error": type(exc).__name__}
        if snap is not None:
            snapshots[name] = snap
    return snapshots


# -------------------------------------------------------------------- rendering

def format_report(results: List[CheckResult]) -> str:
    """CLI rendering: one aligned line per check."""
    if not results:
        return "(no checks ran)"
    width = max(len(r.name) for r in results)
    lines = [f"  {r.status.value.upper():5} {r.name:{width}}  {r.message}"
             for r in results]
    counts = _count(results)
    lines.append("")
    lines.append("  " + ", ".join(f"{n} {s}" for s, n in counts.items() if n))
    return "\n".join(lines)


def summarize(results: List[CheckResult]) -> str:
    """One spoken-style sentence for the assistant to say."""
    if not results:
        return "I could not run any self-diagnostics, sir."
    counts = _count(results)
    total = len(results)
    bad = [r for r in results if r.status is CheckStatus.FAIL]
    warn = [r for r in results if r.status is CheckStatus.WARN]
    def _n(n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"

    head = (f"{total} checks: {counts['pass']} passed"
            + (f", {_n(counts['warn'], 'warning')}" if counts["warn"] else "")
            + (f", {_n(counts['fail'], 'failure')}" if counts["fail"] else "")
            + (f", {counts['skip']} skipped" if counts["skip"] else ""))
    if not bad and not warn:
        return f"All systems nominal, sir — {head}."
    worst = bad[0] if bad else warn[0]
    return f"{head}. Most notable: {worst.name} — {worst.message}"


def _count(results: List[CheckResult]) -> Dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for r in results:
        counts[r.status.value] += 1
    return counts
