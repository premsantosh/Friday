"""
Sandboxed third-party bot fallback (M7) — the riskiest channel, off by default.

When our own GenericWeb engine can't book a site, we *may* fall back to a
community automation bot from GitHub. Running third-party code is dangerous, so
this is gated hard:

  - Disabled unless RESERVATION_ALLOW_SANDBOX_BOTS is set AND Docker is present.
  - The candidate repo is vetted (stars / recent activity / allowed language).
  - It runs only inside a locked-down Docker container: no host mounts, no host
    env/secrets, dropped capabilities, read-only FS, pids/memory limits, a hard
    timeout, and a restricted network. The container clones the repo itself, so
    nothing from the host filesystem is exposed. Container + clone are torn down.
  - The workflow requires a SEPARATE, repo-named consent before this ever runs.

Honest limits: generically driving an arbitrary repo to *complete* a booking is
not reliably solvable or fully safe. This harness runs best-effort and reports a
hand-off when it can't confirm success. The finder and sandbox runner are
injectable so the orchestration/gating is testable without GitHub or Docker.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from core.harness import Sink, SinkMode, guard

logger = logging.getLogger(__name__)

# Untrusted code gets booking facts only — no guest PII, no credentials.
SANDBOX_SINK = Sink("sandbox", SinkMode.ALLOWLIST,
                    frozenset({"business_name", "url", "date", "time", "party_size"}))

ALLOWED_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Go", "Ruby"}
MIN_STARS = 5
MAX_AGE_DAYS = 730  # repo must have been pushed within ~2 years


@dataclass
class BotCandidate:
    full_name: str        # "owner/repo"
    url: str
    stars: int
    language: str
    pushed_at: str        # ISO date


@dataclass
class SandboxResult:
    success: bool
    output: str = ""
    confirmation: Optional[str] = None
    error: Optional[str] = None


class GitHubBotFinder:
    """Searches GitHub (via `gh`) and vets a candidate bot. Injectable searcher for tests."""

    def __init__(self, searcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None):
        self._searcher = searcher

    def find(self, query: str) -> Optional[BotCandidate]:
        for repo in self._search(query):
            cand = self._vet(repo)
            if cand is not None:
                return cand
        return None

    def _search(self, query: str) -> List[Dict[str, Any]]:
        if self._searcher is not None:
            return self._searcher(query)
        try:
            proc = subprocess.run(
                ["gh", "search", "repos", query, "--limit", "10",
                 "--json", "fullName,url,stargazersCount,language,pushedAt,isArchived"],
                capture_output=True, text=True, timeout=20, check=True,
            )
            return json.loads(proc.stdout or "[]")
        except Exception:
            logger.warning("GitHub bot search failed.", exc_info=True)
            return []

    @staticmethod
    def _vet(repo: Dict[str, Any]) -> Optional[BotCandidate]:
        if repo.get("isArchived"):
            return None
        stars = int(repo.get("stargazersCount", 0))
        lang = repo.get("language") or ""
        if stars < MIN_STARS or lang not in ALLOWED_LANGUAGES:
            return None
        pushed = repo.get("pushedAt", "")
        try:
            dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - dt > timedelta(days=MAX_AGE_DAYS):
                return None
        except (ValueError, AttributeError):
            return None
        return BotCandidate(full_name=repo.get("fullName", ""), url=repo.get("url", ""),
                            stars=stars, language=lang, pushed_at=pushed)


class DockerSandbox:
    """Runs a repo inside a hardened, throwaway container. Injectable runner for tests."""

    def __init__(self, runner: Optional[Callable] = None,
                 image: str = "python:3.12-slim", network: str = "none", timeout: int = 180):
        self._runner = runner
        self.image = image
        self.network = network   # "none" by default; a domain-filtered proxy net is the real fix
        self.timeout = timeout

    def run(self, repo_url: str, payload: Dict[str, Any]) -> SandboxResult:
        if self._runner is not None:
            return self._runner(repo_url, payload)
        return self._run_docker(repo_url, payload)

    def _run_docker(self, repo_url: str, payload: Dict[str, Any]) -> SandboxResult:
        # The container clones the repo itself (no host mount) and runs a conventional
        # entrypoint, with the booking payload on stdin. No host env/secrets are passed.
        script = (
            "set -e; git clone --depth 1 "
            f"{_shell_quote(repo_url)} /tmp/bot >/dev/null 2>&1; cd /tmp/bot; "
            "if [ -f requirements.txt ]; then pip install -q -r requirements.txt || true; fi; "
            "if [ -f main.py ]; then python main.py; "
            "elif [ -f package.json ]; then npm -s start; "
            "else echo __NO_ENTRYPOINT__; fi"
        )
        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", self.network,
            "--read-only", "--tmpfs", "/tmp:exec",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--memory", "512m", "--cpus", "1",
            "--user", "nobody",
            "--env-file", "/dev/null",          # never inherit host env/secrets
            self.image, "sh", "-c", script,
        ]
        try:
            proc = subprocess.run(cmd, input=json.dumps(payload), capture_output=True,
                                  text=True, timeout=self.timeout)
            out = (proc.stdout or "")[:4000]
            if "__NO_ENTRYPOINT__" in out or proc.returncode != 0:
                return SandboxResult(False, output=out, error="no_entrypoint_or_failed")
            low = out.lower()
            success = "confirm" in low or "booked" in low or "reservation" in low
            return SandboxResult(success, output=out, confirmation=_find_conf(out))
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error="timeout")
        except Exception as exc:
            logger.warning("Docker sandbox run failed.", exc_info=True)
            return SandboxResult(False, error=str(exc))


class SandboxBotChannel:
    """Finds and runs a vetted third-party bot in a sandbox. Off unless explicitly enabled."""

    requires_consent = True

    def __init__(self, finder: GitHubBotFinder, sandbox: DockerSandbox):
        self.finder = finder
        self.sandbox = sandbox

    @classmethod
    def from_env(cls) -> Optional["SandboxBotChannel"]:
        if os.getenv("RESERVATION_ALLOW_SANDBOX_BOTS", "").lower() not in ("1", "true", "yes"):
            return None
        if shutil.which("docker") is None:
            logger.info("Sandbox bots enabled but Docker not found; disabling.")
            return None
        return cls(GitHubBotFinder(), DockerSandbox())

    def find_bot(self, business: str, platform: str) -> Optional[BotCandidate]:
        return self.finder.find(f"{platform} reservation bot")

    def run_bot(self, candidate: BotCandidate, details: Dict[str, Any]) -> SandboxResult:
        # Pass only the minimum booking facts — never guest PII, credentials,
        # or host env. The sink check makes the minimization structural.
        payload = {k: details.get(k) for k in
                   ("business_name", "url", "date", "time", "party_size")}
        guard(SANDBOX_SINK, payload)
        return self.sandbox.run(candidate.url, payload)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _find_conf(text: str) -> Optional[str]:
    import re
    m = re.search(r"(?i)confirmation\s*(?:#|number|code)?[:\s]*([A-Z0-9\-]{4,})", text)
    return m.group(1) if m else None
