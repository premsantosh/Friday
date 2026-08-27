"""Self-awareness workflows: `self_status` (read-only) and `self_repair` (gated).

`self_status` is how Friday answers questions about itself — "did the LoRA run
last night?", "what did you do today?", "run a self-diagnosis", "what can you
do?" — by reading the introspection providers (introspection/), the live
workflow registry, and the LLM's own counters. It is registered on both
engines: the legacy router routes to it like any workflow, and the agent
engine auto-generates a `self_status` tool from it.

`self_repair` is the v1 groundwork for fixing itself: a small allowlist of
safe operations (re-run the nightly learning cycle, revert an artifact
pointer), each behind an explicit spoken confirmation and executed through the
harness ActionGate — kill switch, approval binding, idempotency, and an
append-only audit trail Friday can later report on. Deliberately out of scope
for v1: anything unattended — clearing locks, restarting channels, pruning
logs, changing its own code.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from workflows.base import (
    ConversationalWorkflow,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
    WorkflowTrigger,
)

logger = logging.getLogger(__name__)

TOPICS = ("overview", "nightly", "model", "activity", "jobs", "health",
          "capabilities")

# intent keyword → topic, first match wins (checked in order).
_TOPIC_HINTS = (
    (("diagnos", "health", "doctor", "check yourself", "self-check",
      "self check"), "health"),
    (("lora", "train", "learn", "nightly", "last night", "fine-tun",
      "fine tun"), "nightly"),
    (("what can you do", "capabilit", "abilities", "skills"), "capabilities"),
    (("what did you do", "what have you done", "activity", "audit",
      "actions"), "activity"),
    (("job", "schedul", "wake", "pending", "task"), "jobs"),
    (("model", "adapter", "version", "engine"), "model"),
)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


class SelfStatusWorkflow(Workflow):
    """Report on Friday's own state. Read-only; safe on every engine and mode."""

    read_only = True

    def __init__(self, workflow_manager=None, paths=None, probes=None):
        self._manager = workflow_manager
        self._paths = paths        # introspection.Paths; None → defaults
        self._probes = probes      # introspection.Probes; None → defaults
        self._llm_stats_fn: Optional[Callable[[], str]] = None
        self._engine_label_fn: Optional[Callable[[], str]] = None
        self._ephemeral = False

    def bind_runtime(self, *, llm_stats_fn=None, engine_label_fn=None,
                     ephemeral: bool = False) -> None:
        """Late wiring from the running assistant (main.py / VoiceAssistant)."""
        self._llm_stats_fn = llm_stats_fn
        self._engine_label_fn = engine_label_fn
        self._ephemeral = ephemeral

    @property
    def name(self) -> str:
        return "self_status"

    @property
    def description(self) -> str:
        return ("Report on Friday's own state: nightly learning runs and LoRA "
                "training, scheduled jobs, recent actions, self-diagnosis, "
                "and what it can do")

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            examples=[
                "Did the LoRA run last night?",
                "What did you do today?",
                "Run a self-diagnosis",
                "What's your status?",
                "What can you do?",
                "Is your nightly training healthy?",
            ]
        )

    # ------------------------------------------------------------------ execute
    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        topic = str(entities.get("topic") or "").strip().lower()
        if topic not in TOPICS:
            topic = self._infer_topic(intent)
        try:
            message, data = self._report(topic)
        except Exception as exc:
            logger.warning("self_status failed for topic %s", topic, exc_info=True)
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message="I attempted to examine myself and ran into a snag, "
                        f"sir — {type(exc).__name__}. My apologies.",
                data={"topic": topic, "error": type(exc).__name__},
            )
        return WorkflowResult(status=WorkflowStatus.SUCCESS, message=message,
                              data={"topic": topic, **data})

    @staticmethod
    def _infer_topic(intent: str) -> str:
        text = intent.lower()
        for needles, topic in _TOPIC_HINTS:
            if any(n in text for n in needles):
                return topic
        return "overview"

    # ------------------------------------------------------------------ reports
    def _report(self, topic: str):
        from introspection import gather_status, run_doctor, summarize

        if topic == "health":
            results = run_doctor(self._paths, self._probes,
                                 workflow_manager=self._manager)
            data = {"checks": [{"name": r.name, "status": r.status.value,
                                "message": r.message} for r in results]}
            return summarize(results), data

        if topic == "capabilities":
            return self._capabilities()

        status = gather_status(self._paths, self._probes,
                               workflow_manager=self._manager)
        handler = {
            "nightly": self._nightly_message,
            "model": self._model_message,
            "activity": self._activity_message,
            "jobs": self._jobs_message,
            "overview": self._overview_message,
        }[topic]
        if topic == "overview":
            data = {"status": status}
        else:
            keys = _SLICES[topic]
            data = {"status": {k: status.get(k) for k in keys}}
        return handler(status), data

    def _capabilities(self):
        if self._manager is None:
            return ("I am not entirely sure what I can do at the moment, sir — "
                    "my capability registry is not attached.", {})
        names = self._manager.list_workflows()
        described = [f"{name} — {wf.description}"
                     for name, wf in self._manager.workflows.items()]
        message = (f"I currently have {_plural(len(names), 'capability')}, sir: "
                   f"{', '.join(sorted(names))}.")
        data: Dict[str, Any] = {"capabilities": described}
        if self._llm_stats_fn is not None:
            try:
                data["session_stats"] = self._llm_stats_fn()
            except Exception:
                pass
        if self._engine_label_fn is not None:
            try:
                data["engine"] = self._engine_label_fn()
            except Exception:
                pass
        return message, data

    def _records_unavailable(self) -> str:
        if self._ephemeral:
            return ("My persistent records are not attached in this mode, sir — "
                    "run me normally and I shall have my full memory of myself.")
        return ("I have no records of that yet, sir — my research substrate "
                "has not produced any.")

    def _nightly_message(self, status: Dict[str, Any]) -> str:
        nightly = status.get("nightly", {})
        if not nightly.get("available"):
            return self._records_unavailable()
        runs = nightly.get("runs") or []
        if not runs:
            return ("No nightly learning run has been recorded yet, sir. "
                    "The schedule expects one daily at 03:30.")
        last = runs[0]
        age = last.get("hours_ago")
        if age is None:
            when = "recently"
        elif age < 1:
            when = "within the last hour"
        else:
            when = f"about {age:g} hours ago"
        parts = []
        if last["failed_stages"]:
            parts.append(f"My last learning run, {when}, ran into trouble, sir — "
                         f"the {', '.join(last['failed_stages'])} "
                         f"stage{'s' if len(last['failed_stages']) > 1 else ''} failed.")
        else:
            stages = last.get("stages") or {}
            parts.append(f"Yes, sir — my last learning run completed {when}, "
                         f"all {_plural(len(stages), 'stage')} in order.")
        lora = (status.get("arms", {}).get("arms") or {}).get("lora")
        if lora:
            if lora.get("gated"):
                parts.append(f"The LoRA adapter trained to {lora['current']} but "
                             f"was held back by the quality gate.")
            elif lora.get("current"):
                parts.append(f"The current LoRA adapter is {lora['current']} "
                             f"({_plural(lora['version_count'], 'version')} on disk).")
            else:
                parts.append("No LoRA adapter has been promoted yet.")
        lock = nightly.get("lock") or {}
        if lock.get("held"):
            parts.append("A run appears to be in progress at this very moment.")
        return " ".join(parts)

    def _model_message(self, status: Dict[str, Any]) -> str:
        parts = []
        if self._engine_label_fn is not None:
            try:
                parts.append(f"I am reasoning with the {self._engine_label_fn()} "
                             f"engine, sir.")
            except Exception:
                pass
        arms = status.get("arms", {}).get("arms") or {}
        if arms:
            for arm, state in sorted(arms.items()):
                current = state.get("current") or "none promoted"
                gated = " (gated)" if state.get("gated") else ""
                parts.append(f"{arm}: {current}{gated}, "
                             f"{_plural(state.get('version_count', 0), 'version')}.")
        else:
            parts.append("No local model artifacts exist yet, sir.")
        if self._llm_stats_fn is not None:
            try:
                parts.append(f"This session: {self._llm_stats_fn()}.")
            except Exception:
                pass
        return " ".join(parts)

    def _activity_message(self, status: Dict[str, Any]) -> str:
        activity = status.get("activity", {})
        if not activity.get("available"):
            return self._records_unavailable()
        parts = []
        routes = activity.get("routes_24h") or {}
        if routes:
            total = sum(routes.values())
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(routes.items()))
            parts.append(f"In the last day I handled {_plural(total, 'exchange')} "
                         f"({detail}), sir.")
        actions = activity.get("gated_actions") or []
        executed = [a for a in actions if a["event"] == "EXEC_OK"]
        denied = [a for a in actions if a["event"] == "GATE_DENY"]
        if executed:
            names = sorted({a["workflow"] for a in executed})
            parts.append(f"I carried out {_plural(len(executed), 'gated action')} "
                         f"({', '.join(names)}).")
        if denied:
            parts.append(f"{_plural(len(denied), 'proposal')} was stopped at the gate.")
        if not parts:
            parts.append("A quiet day so far, sir — nothing of note in my records.")
        return " ".join(parts)

    def _jobs_message(self, status: Dict[str, Any]) -> str:
        jobs = status.get("jobs", {})
        parts = []
        launchd = jobs.get("launchd")
        if launchd is None:
            parts.append("I cannot see the system scheduler from here, sir.")
        elif launchd:
            parts.append(f"Scheduled jobs loaded: {', '.join(launchd)}.")
        else:
            parts.append("No scheduled jobs of mine are loaded, sir.")
        wakes = jobs.get("pending_wakes") or []
        if wakes:
            parts.append(f"I have {_plural(len(wakes), 'wake-up')} pending.")
        sessions = jobs.get("live_sessions") or []
        if sessions:
            n = sum(s["count"] for s in sessions)
            parts.append(f"{_plural(n, 'task session')} in flight.")
        if len(parts) == 1 and not wakes and not sessions:
            parts.append("Nothing else is pending.")
        return " ".join(parts)

    def _overview_message(self, status: Dict[str, Any]) -> str:
        parts = []
        nightly = status.get("nightly", {})
        if nightly.get("available") and nightly.get("runs"):
            last = nightly["runs"][0]
            if last["failed_stages"]:
                parts.append(f"My last learning run failed at "
                             f"{', '.join(last['failed_stages'])}, sir.")
            else:
                parts.append("My last learning run completed cleanly, sir.")
        else:
            parts.append("I have no learning-run records in this mode, sir.")
        storage = status.get("storage", {})
        if storage.get("available"):
            parts.append(f"{_plural(len(storage.get('databases', {})), 'database')} "
                         f"on disk, {storage.get('disk_free_gb', '?')} GB free.")
        if self._manager is not None:
            parts.append(f"{_plural(len(self._manager.list_workflows()), 'capability')} "
                         f"registered.")
        parts.append("Ask for a self-diagnosis if you would like the full picture.")
        return " ".join(parts)


# Which gather_status keys back each topic's structured data slice.
_SLICES = {"nightly": ("nightly", "arms"), "model": ("arms",),
           "activity": ("activity",), "jobs": ("jobs",)}


# ============================================================== self-repair

def _default_launchctl_kickstart() -> subprocess.CompletedProcess:
    if shutil.which("launchctl") is None:
        raise FileNotFoundError("launchctl not available on this platform")
    return subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.friday.nightly"],
        capture_output=True, text=True, timeout=15)


class SelfRepairWorkflow(ConversationalWorkflow):
    """Confirmation-gated self-repair. v1 allowlist:

      rerun_nightly    — kick the launchd nightly job now (the flock in
                         cmd_nightly makes an accidental double-fire a no-op)
      revert_artifact  — repoint an arm's `current` at an earlier version

    Every execution goes through ActionGate.with_defaults(): the kill switch
    is honoured, the user's YES is bound to the exact plan shown, replays are
    idempotent, and the audit log records what actually happened — which is
    also how `self_status` can later report on it.
    """

    session_timeout_s = 300
    read_only = False

    def __init__(self, launchctl_kickstart=None, artifacts_dir=None,
                 db_path: str = "~/.friday/research.db"):
        self._kickstart = launchctl_kickstart or _default_launchctl_kickstart
        self._artifacts_dir = artifacts_dir
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "self_repair"

    @property
    def description(self) -> str:
        return ("Repair Friday itself (with confirmation): re-run the nightly "
                "learning cycle, or revert a model artifact to an earlier version")

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            examples=[
                "Re-run the nightly training",
                "Kick off last night's learning run again",
                "Revert the LoRA model to the previous version",
                "Roll back your adapter to v20260718",
            ]
        )

    # -------------------------------------------------------------------- start
    async def start(self, intent: str, entities: Dict[str, Any], session) -> "TurnResult":
        from core.conversation.session import TurnResult

        plan = self._parse_plan(intent, entities)
        if plan is None:
            return TurnResult.ask(
                "Which repair would you like, sir: re-run the nightly learning "
                "cycle, or revert a model artifact to an earlier version?",
                next_state="choose")
        if plan["action"] == "revert_artifact" and not plan.get("to"):
            return TurnResult.ask(self._version_prompt(plan["arm"]),
                                  slots_update={"plan": plan},
                                  next_state="need_version")
        return TurnResult.confirm(self._question(plan),
                                  slots_update={"plan": plan},
                                  next_state="confirm_repair")

    # ------------------------------------------------------------------- resume
    async def resume(self, text: str, session) -> "TurnResult":
        from core.conversation.session import TurnResult
        from core.harness.confirm import ConfirmDecision, parse_confirmation

        if session.fsm_state == "choose":
            plan = self._parse_plan(text, {})
            if plan is None:
                return TurnResult.cancel(
                    "Very well, sir — I shall leave myself as I am.")
            if plan["action"] == "revert_artifact" and not plan.get("to"):
                return TurnResult.ask(self._version_prompt(plan["arm"]),
                                      slots_update={"plan": plan},
                                      next_state="need_version")
            return TurnResult.confirm(self._question(plan),
                                      slots_update={"plan": plan},
                                      next_state="confirm_repair")

        if session.fsm_state == "need_version":
            plan = dict(session.slots.get("plan") or {})
            version = self._extract_version(text)
            if version is None:
                return TurnResult.cancel(
                    "I could not make out a version, sir — no changes made.")
            plan["to"] = version
            return TurnResult.confirm(self._question(plan),
                                      slots_update={"plan": plan},
                                      next_state="confirm_repair")

        # confirm_repair
        decision = parse_confirmation(text)
        if decision == ConfirmDecision.NO:
            return TurnResult.cancel("Very well, sir — no changes made.")
        if decision != ConfirmDecision.YES:
            plan = session.slots.get("plan") or {}
            return TurnResult.confirm(
                f"A simple yes or no, sir — {self._question(plan)}")
        return await self._execute_plan(session)

    # ---------------------------------------------------------------- execution
    async def _execute_plan(self, session) -> "TurnResult":
        from core.conversation.session import TurnResult
        from core.harness import Action, ActionGate, ActionKind
        from core.harness.audit import AuditLog

        plan = session.slots.get("plan") or {}
        action = Action(kind=ActionKind.SELF_REPAIR,
                        session_id=session.session_id,
                        workflow=self.name, plan=plan)
        gate = ActionGate.with_defaults(kill_switch_env="FRIDAY_KILL_SWITCH",
                                        audit=AuditLog.from_env())
        gate.record_approval(session, action)
        outcome = await gate.execute(
            action, session, lambda: self._run(plan),
            success_of=lambda r: bool(r and r.get("success")))
        if not outcome.ok:
            return TurnResult.complete(
                f"I must decline, sir — {outcome.refusal.detail}")
        result = outcome.result or {}
        if result.get("success"):
            return TurnResult.complete(result.get("message", "Done, sir."))
        return TurnResult.complete(
            result.get("message", "The repair did not take, sir."))

    async def _run(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        action = plan.get("action")
        if action == "rerun_nightly":
            try:
                proc = self._kickstart()
            except Exception as exc:
                return {"success": False,
                        "message": f"I could not reach the scheduler, sir — "
                                   f"{type(exc).__name__}. The nightly can be run "
                                   f"manually with `python -m research nightly`."}
            if proc.returncode == 0:
                return {"success": True,
                        "message": "The nightly learning cycle is starting now, "
                                   "sir. I shall have fresh results shortly."}
            return {"success": False,
                    "message": "The scheduler declined to start the run, sir — "
                               f"launchctl exited {proc.returncode}."}

        if action == "revert_artifact":
            from research.ops import revert_arm

            try:
                previous = revert_arm(plan["arm"], plan["to"],
                                      artifacts_dir=self._artifacts_dir,
                                      db_path=self._db_path, via="self_repair")
            except ValueError as exc:
                return {"success": False, "message": f"{exc}, sir."}
            return {"success": True,
                    "message": f"Done, sir — {plan['arm']} is back on "
                               f"{plan['to']} (was {previous or 'unset'})."}

        return {"success": False, "message": "I do not know that repair, sir."}

    # ------------------------------------------------------------------ parsing
    def _parse_plan(self, text: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(entities.get("action") or "").strip().lower()
        lowered = text.lower()
        if not action:
            if "revert" in lowered or "roll back" in lowered or "rollback" in lowered:
                action = "revert_artifact"
            elif any(w in lowered for w in ("nightly", "training", "learning",
                                            "run", "cycle")):
                action = "rerun_nightly"
        if action == "rerun_nightly":
            return {"action": "rerun_nightly"}
        if action == "revert_artifact":
            arm = str(entities.get("arm") or "").strip().lower()
            if not arm:
                arm = next((a for a in self._known_arms() if a in lowered), "lora")
            version = (str(entities.get("version") or "").strip()
                       or self._extract_version(text) or "")
            return {"action": "revert_artifact", "arm": arm, "to": version}
        return None

    @staticmethod
    def _extract_version(text: str) -> Optional[str]:
        import re

        match = re.search(r"\bv\d{8}(?:-\d+)?\b", text)
        if match:
            return match.group(0)
        if text.strip().lower() in ("previous", "the previous version",
                                    "previous version", "last one"):
            return None  # resolved in _version_prompt flow by explicit choice
        return None

    def _known_arms(self) -> List[str]:
        try:
            from introspection.providers import discover_arms
            from research.artifacts import DEFAULT_ARTIFACTS_DIR
            from pathlib import Path

            art_dir = (Path(self._artifacts_dir).expanduser()
                       if self._artifacts_dir else DEFAULT_ARTIFACTS_DIR)
            if art_dir.exists():
                return discover_arms(art_dir)
        except Exception:
            pass
        return ["lora", "memory", "prompt"]

    def _versions_of(self, arm: str) -> List[str]:
        try:
            from research import artifacts
            from pathlib import Path

            art_dir = (Path(self._artifacts_dir).expanduser()
                       if self._artifacts_dir else artifacts.DEFAULT_ARTIFACTS_DIR)
            return artifacts.list_versions(arm, art_dir)
        except Exception:
            return []

    def _version_prompt(self, arm: str) -> str:
        versions = self._versions_of(arm)
        if versions:
            shown = ", ".join(versions[-5:])
            return (f"Which version of {arm} shall I revert to, sir? "
                    f"On disk: {shown}.")
        return f"Which version of {arm} shall I revert to, sir?"

    def _question(self, plan: Dict[str, Any]) -> str:
        if plan.get("action") == "rerun_nightly":
            return ("Shall I re-run last night's learning cycle now, sir? "
                    "It takes a few minutes and happens in the background.")
        if plan.get("action") == "revert_artifact":
            return (f"Shall I revert the {plan.get('arm')} artifact to "
                    f"{plan.get('to')}, sir?")
        return "Shall I proceed, sir?"
