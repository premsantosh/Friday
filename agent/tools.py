"""
Tools for the agent graph, generated from the existing WorkflowManager registry.
No per-workflow code:

  * Simple workflows → one StructuredTool each. Name = workflow.name,
    description = workflow.description + trigger examples (the same text the
    legacy IntentRouter put in its prompt, so routing quality carries over).
    A WorkflowStatus.FAILURE comes back as "ERROR: …" so the model apologises
    in personality (replaces VoiceAssistant._handle_workflow_failure on this path).

  * Conversational workflows → a single `start_task(workflow_name, intent,
    entities)` handoff tool. It opens a legacy session (exactly what
    VoiceAssistant._maybe_start_session does) and sets `handoff_message` so the
    graph ends the turn and the engine returns the session's opening line
    VERBATIM. Subsequent turns route into the legacy session at Step 0; the
    graph is not involved until that session closes.

  * Gated tools (Phase 3) for actions that need a confirmation — see
    `gated_workflow_tool` and GATE_SPECS (today: `hass_locks` unlock; other
    workflows are bound ungated, exactly as the legacy router executes them).
    INVARIANTS: the model never holds a raw callable — every tool is a wrapper
    around a registered Workflow; everything before `interrupt()` is pure
    (LangGraph re-executes the tool function on resume); the post-resume
    execution goes through ActionGate (kill switch, idempotency via the audit
    log keyed by the tool call id) so a replay can never double-fire.

  * `schedule_wakeup(delay_minutes, note)` (Phase 4) — writes an agent_wakes row
    that BackgroundTaskRunner services by re-invoking this user's thread.

Tool calls run one at a time (`parallel_tool_calls=False` where the provider
supports it) so an interrupt in one tool can't re-execute a side-effecting
sibling on resume.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from core.harness import Action, ActionGate, ActionKind, hash_plan
from workflows import ConversationalWorkflow, WorkflowStatus

logger = logging.getLogger(__name__)

ERROR_PREFIX = "ERROR:"
DECLINED_PREFIX = "DECLINED:"
REFUSED_PREFIX = "REFUSED:"
# Prefixes that mean "the action did not happen" — never cache/learn from these.
FAILURE_PREFIXES = (ERROR_PREFIX, DECLINED_PREFIX, REFUSED_PREFIX, "Error:")

START_TASK = "start_task"
SCHEDULE_WAKEUP = "schedule_wakeup"

MAX_WAKE_MINUTES = 7 * 24 * 60


def safe_tool_name(name: str) -> str:
    """Provider tool names must match ^[a-zA-Z0-9_-]{1,64}$."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "workflow"


def tool_description(workflow) -> str:
    examples = ", ".join(f'"{ex}"' for ex in workflow.trigger.examples[:3])
    desc = (workflow.description or workflow.name).strip().rstrip(".")
    return f"{desc}. Examples: {examples}" if examples else desc


def format_result(result) -> str:
    """WorkflowResult → tool message text."""
    if result is None:
        return f"{ERROR_PREFIX} no result"
    if result.status == WorkflowStatus.FAILURE:
        return f"{ERROR_PREFIX} {result.error or result.message or 'the action failed'}"
    return result.message or result.status.value


# ------------------------------------------------------------------- gating

@dataclass(frozen=True)
class GateSpec:
    """How a workflow's calls are gated: `requires_confirmation(intent,
    entities)` decides per call (pure), `question` phrases the confirmation."""
    kind: ActionKind
    requires_confirmation: Callable[[str, Dict[str, Any]], bool]
    question: Callable[[str, Dict[str, Any]], str]


def _unlock_requested(intent: str, entities: Dict[str, Any]) -> bool:
    action = str(entities.get("action", "")).lower()
    return action == "unlock" or (not action and "unlock" in intent.lower())


def door_name(entities: Dict[str, Any]) -> str:
    """'back' / 'back door' / 'the back door' → 'back' (models vary in what
    they put in the entity; the question must not read 'back door door')."""
    raw = str(entities.get("door") or "front").strip().lower()
    raw = re.sub(r"^(the|my)\s+", "", raw)
    raw = re.sub(r"\s*door$", "", raw)
    return raw or "front"


# Proving ground: Home Assistant door locks — unlocking needs a confirmation;
# lock / status pass straight through.
GATE_SPECS: Dict[str, GateSpec] = {
    "hass_locks": GateSpec(
        kind=ActionKind.DEVICE_CONTROL,
        requires_confirmation=_unlock_requested,
        question=lambda intent, e: f"Shall I unlock the {door_name(e)} door, sir?",
    ),
}


class _GateSession:
    """Duck-typed stand-in for core.conversation.Session: ActionGate only needs
    `.slots` (approvals live there) and `.fsm_state`. One per gated tool call,
    keyed by the tool_call_id so the audit trail / idempotency key is stable
    across a resume replay and distinct from any other proposal."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.slots: Dict[str, Any] = {}
        self.fsm_state = "agent_confirm"


# ------------------------------------------------------------- tool makers

def workflow_to_tool(workflow) -> StructuredTool:
    async def _run(intent: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """Run the workflow.

        Args:
            intent: The user's request in their own words.
            entities: Extracted parameters (for example device, room, action,
                door, temperature, duration). Empty when there are none.
        """
        result = await workflow.execute(intent, dict(entities or {}))
        return format_result(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name=safe_tool_name(workflow.name),
        description=tool_description(workflow),
        parse_docstring=True,
    )


def gated_workflow_tool(workflow, spec: GateSpec, gate: ActionGate,
                        clock: Callable[[], float] = time.time) -> StructuredTool:
    async def _run(
        intent: str,
        entities: Optional[Dict[str, Any]] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Run the workflow; some actions ask the user to confirm first.

        Args:
            intent: The user's request in their own words.
            entities: Extracted parameters (for example device, room, action,
                door, temperature, duration). Empty when there are none.
        """
        entities = dict(entities or {})
        if not spec.requires_confirmation(intent, entities):
            return format_result(await workflow.execute(intent, entities))

        # ---- pure prologue: no side effects above interrupt() — this code runs
        # again from the top when the graph resumes.
        plan = {"workflow": workflow.name, "intent": intent, "entities": entities}
        payload = {
            "type": "confirmation",
            "question": spec.question(intent, entities),
            "plan": plan,
            "plan_hash": hash_plan(plan),
            "tool_call_id": tool_call_id,
            "asked_at": clock(),   # engine compares with the same clock for soft expiry
        }
        answer = interrupt(payload)
        decision = answer.get("decision") if isinstance(answer, dict) else str(answer)
        if decision != "yes":
            return f"{DECLINED_PREFIX} the user declined; do not retry and do not ask again."

        # ---- post-resume: the user said yes to exactly `plan` (same dict, same
        # hash). Note the approval is recorded and consumed right here, so
        # ApprovalPolicy is satisfied by construction on this path — the real
        # protection is interrupt() + the deterministic yes, the kill switch
        # (checked now, at execution time) and IdempotencyPolicy + audit trail.
        session = _GateSession(f"agent:{tool_call_id or hash_plan(plan)[:12]}")
        action = Action(kind=spec.kind, session_id=session.session_id,
                        workflow=workflow.name, plan=plan)
        gate.record_approval(session, action)
        outcome = await gate.execute(
            action, session, lambda: workflow.execute(intent, entities),
            success_of=lambda r: r is not None and r.status == WorkflowStatus.SUCCESS)
        if not outcome.ok:
            return f"{REFUSED_PREFIX} {outcome.refusal.detail}"
        return format_result(outcome.result)

    return StructuredTool.from_function(
        coroutine=_run,
        name=safe_tool_name(workflow.name),
        description=tool_description(workflow),
        parse_docstring=True,
    )


def make_start_task_tool(conversational: Dict[str, Any], sessions, context) -> StructuredTool:
    names = ", ".join(conversational)
    listing = "\n".join(f"- {name}: {tool_description(wf)}" for name, wf in conversational.items())

    async def _run(
        workflow_name: str,
        intent: str,
        entities: Optional[Dict[str, Any]] = None,
        user_id: Annotated[str, InjectedState("user_id")] = "default",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ):
        """Hand the conversation to a multi-turn task. After calling this, stop.

        Args:
            workflow_name: Which task to start.
            intent: The user's request in their own words.
            entities: Anything already known (names, dates, times, party size…).
        """
        wf = conversational.get(workflow_name)
        if wf is None:
            return f"{ERROR_PREFIX} unknown task '{workflow_name}'. Available tasks: {names}"
        if sessions is None:
            return f"{ERROR_PREFIX} multi-turn tasks are disabled"
        if sessions.has_active(user_id):
            return f"{ERROR_PREFIX} a task is already in progress for this user"
        turn = await sessions.open(wf, intent, dict(entities or {}), user_id)
        if context is not None:
            context.update(wf.name, dict(entities or {}), intent)
        message = turn.message or ""
        return Command(update={
            "handoff_message": message,
            "messages": [ToolMessage(content=message or "(task started)", tool_call_id=tool_call_id)],
        })

    return StructuredTool.from_function(
        coroutine=_run,
        name=START_TASK,
        description=("Start a multi-turn task that will ask the user follow-up questions "
                     "(bookings, reminders, anything needing several pieces of information). "
                     f"Tasks:\n{listing}"),
        parse_docstring=True,
    )


def make_schedule_wakeup_tool(store, clock: Callable[[], float] = time.time) -> StructuredTool:
    async def _run(
        delay_minutes: float,
        note: str,
        user_id: Annotated[str, InjectedState("user_id")] = "default",
    ) -> str:
        """Schedule a wake-up for yourself to follow up later (re-check something,
        remind the user, finish a deferred step). When it fires you receive the
        note and can act on it.

        Args:
            delay_minutes: How many minutes from now to wake up (1 to 10080).
            note: What to do when woken, in one sentence.
        """
        try:
            minutes = float(delay_minutes)
        except (TypeError, ValueError):
            return f"{ERROR_PREFIX} delay_minutes must be a number"
        minutes = max(1.0, min(float(MAX_WAKE_MINUTES), minutes))
        wake_at = clock() + minutes * 60
        store.add_wake(user_id, wake_at, {"note": note})
        return f"Wake-up scheduled in {minutes:g} minutes: {note}"

    return StructuredTool.from_function(
        coroutine=_run,
        name=SCHEDULE_WAKEUP,
        description=("Schedule a wake-up for yourself N minutes from now to follow up on "
                     "something (re-check a state, remind the user, finish a deferred step)."),
        parse_docstring=True,
    )


# ---------------------------------------------------------------- registry

@dataclass
class ToolSet:
    tools: List[BaseTool] = field(default_factory=list)
    # tool name -> workflow (simple, single-shot workflows incl. gated ones)
    workflow_for: Dict[str, Any] = field(default_factory=dict)
    simple_names: set = field(default_factory=set)      # ungated simple workflows
    gated_names: set = field(default_factory=set)       # gated wrappers (never cached)
    conversational_names: set = field(default_factory=set)

    def by_name(self, name: str) -> Optional[BaseTool]:
        for t in self.tools:
            if t.name == name:
                return t
        return None


def build_tools(workflows, *, sessions=None, context=None, gate: Optional[ActionGate] = None,
                store=None, gate_specs: Optional[Dict[str, GateSpec]] = None,
                clock: Callable[[], float] = time.time) -> ToolSet:
    """Generate the tool set from a WorkflowManager. `gate_specs` defaults to
    GATE_SPECS; pass {} to disable gating (tests)."""
    specs = GATE_SPECS if gate_specs is None else gate_specs
    ts = ToolSet()
    conversational: Dict[str, Any] = {}
    for name, wf in workflows.workflows.items():
        if isinstance(wf, ConversationalWorkflow):
            conversational[name] = wf
            continue
        spec = specs.get(name)
        if spec is not None and gate is not None:
            tool = gated_workflow_tool(wf, spec, gate, clock=clock)
            ts.gated_names.add(tool.name)
        else:
            tool = workflow_to_tool(wf)
            ts.simple_names.add(tool.name)
        ts.tools.append(tool)
        ts.workflow_for[tool.name] = wf
    if conversational and sessions is not None:
        ts.tools.append(make_start_task_tool(conversational, sessions, context))
        ts.conversational_names = set(conversational)
    if store is not None:
        ts.tools.append(make_schedule_wakeup_tool(store, clock))
    return ts
