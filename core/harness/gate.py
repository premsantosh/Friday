"""
ActionGate — the single door every irreversible, outward-facing action must
pass through (harness spec §3).

The LLM (and the dialogue code around it) only ever *proposes* actions; this
gate is deterministic code that decides whether one may execute, at the moment
of execution:

  - `Action.plan_hash` fingerprints the exact plan the user was shown. The
    user's "yes" is recorded against that hash (`record_approval`), persisted
    in the session's slots so it survives restarts, and `ApprovalPolicy`
    refuses any action whose hash doesn't match a live approval — "what was
    committed is what was approved" is a checked invariant, not a convention.
  - `KillSwitchPolicy` re-reads its env var when the action fires, so a flip
    while a session is WAITING blocks calls/emails/bookings everywhere.
  - `IdempotencyPolicy` consults the audit log: an action that already started
    or succeeded is never executed twice (a crash replay returns a refusal,
    not a second booking).
  - `ConsentScopePolicy` binds consent for untrusted code to the exact repo.
  - `SpendCapPolicy` enforces the hard card cap regardless of what any
    service would be willing to mint.

Workflows may add policies but cannot remove the defaults — `with_defaults()`
is the only public constructor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Mapping, Optional, Sequence, TypeVar

from .audit import EXEC_FAIL, EXEC_OK, EXEC_STARTED, GATE_DENY, AuditLog
from .egress import EgressViolation

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Plan fields that may appear in an audit summary. Never the whole plan.
SUMMARY_FIELDS = ("business_name", "date", "time", "party_size")

# Default approval lifetime. Single-use unless the workflow grants bounded
# retries (e.g. phone redials of an approved call plan).
DEFAULT_APPROVAL_TTL_S = 24 * 3600

_HARNESS_KEY = "_harness"


class ActionKind(Enum):
    BOOK = "book"
    PLACE_CALL = "place_call"
    SEND_EMAIL = "send_email"
    MINT_CARD = "mint_card"
    RUN_UNTRUSTED_CODE = "run_untrusted_code"
    DEVICE_CONTROL = "device_control"   # physical-world device actions gated by the agent engine (e.g. unlock)
    SELF_REPAIR = "self_repair"         # Friday operating on its own state (workflows/introspection.py)


def hash_plan(plan: Mapping[str, Any]) -> str:
    """Canonical fingerprint of a plan: sha256 over sorted, compact JSON."""
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    session_id: str
    workflow: str
    plan: Mapping[str, Any]                 # JSON-serializable; exactly what the user saw
    scope: Optional[str] = None             # consent scope (e.g. "owner/repo")
    amount_usd: Optional[float] = None      # for MINT_CARD
    attempt: int = 0                        # retry counter; part of the idempotency
                                            # key but NOT of the approved plan hash

    @property
    def plan_hash(self) -> str:
        return hash_plan(self.plan)

    def summary(self) -> str:
        parts = [f"{k}={self.plan[k]}" for k in SUMMARY_FIELDS if self.plan.get(k)]
        return " ".join(parts)


@dataclass
class Refusal:
    policy: str
    code: str    # kill_switch | no_approval | scope_mismatch | over_cap | duplicate
    detail: str


@dataclass
class GateOutcome(Generic[T]):
    """Either the executor's result, or the refusal that stopped it."""
    result: Optional[T] = None
    refusal: Optional[Refusal] = None

    @property
    def ok(self) -> bool:
        return self.refusal is None


@dataclass
class GateContext:
    session: Any            # core.conversation.session.Session (duck-typed: .slots)
    audit: AuditLog
    now: float


# ----------------------------------------------------------------- approvals

def _approvals(session) -> List[Dict[str, Any]]:
    return session.slots.setdefault(_HARNESS_KEY, {}).setdefault("approvals", [])


def find_live_approval(session, action: Action, now: Optional[float] = None,
                       gate_states: Optional[frozenset] = None) -> Optional[Dict[str, Any]]:
    """The first approval that is unexpired, not used up, matches the action's
    plan hash — and, when gate states are declared, was recorded in one."""
    now = time.time() if now is None else now
    for a in _approvals(session):
        if (a.get("plan_hash") == action.plan_hash
                and a.get("kind") == action.kind.value
                and a.get("uses", 0) < a.get("max_uses", 1)
                and now < a.get("expires_at", 0)
                and (gate_states is None or a.get("state") in gate_states)):
            return a
    return None


# ------------------------------------------------------------------ policies

class Policy:
    name: str = "policy"

    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]:
        raise NotImplementedError


class KillSwitchPolicy(Policy):
    name = "kill_switch"

    def __init__(self, env_var: str):
        self.env_var = env_var

    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]:
        if os.getenv(self.env_var):
            return Refusal(self.name, "kill_switch",
                           f"{self.env_var} is set; all commits are blocked.")
        return None


class ApprovalPolicy(Policy):
    """A live (unexpired, not used up) approval matching the action's exact
    plan hash must exist. When the workflow declares gate states (FSM, §6),
    the approval must also have been recorded while in one of them."""

    name = "approval"

    def __init__(self, gate_states: Optional[frozenset] = None):
        self.gate_states = gate_states

    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]:
        approval = find_live_approval(ctx.session, action, ctx.now,
                                      gate_states=self.gate_states)
        if approval is None:
            return Refusal(self.name, "no_approval",
                           "No live approval matches this plan (recorded in a gate "
                           "state); it changed or expired.")
        return None


class ConsentScopePolicy(Policy):
    """Untrusted code runs only under a consent naming that exact scope."""

    name = "consent_scope"

    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]:
        if action.kind != ActionKind.RUN_UNTRUSTED_CODE:
            return None
        approval = find_live_approval(ctx.session, action, ctx.now)
        if approval is not None and approval.get("scope") != action.scope:
            return Refusal(self.name, "scope_mismatch",
                           f"Consent was for {approval.get('scope')!r}, "
                           f"not {action.scope!r}.")
        return None


class SpendCapPolicy(Policy):
    name = "spend_cap"
    HARD_CAP_USD = 10.0

    def __init__(self, cap_usd: float = HARD_CAP_USD):
        # A configured cap may only ever lower the ceiling.
        self.cap_usd = min(float(cap_usd), self.HARD_CAP_USD)

    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]:
        if action.kind != ActionKind.MINT_CARD:
            return None
        if action.amount_usd is None or action.amount_usd > self.cap_usd:
            return Refusal(self.name, "over_cap",
                           f"Card amount {action.amount_usd!r} exceeds the "
                           f"${self.cap_usd:.2f} cap (or is unstated).")
        return None


class IdempotencyPolicy(Policy):
    """One execution per (session, kind, plan, attempt). EXEC_FAIL permits a
    retry; EXEC_STARTED with no terminal event means a crash mid-execution —
    refuse rather than risk a double-fire."""

    name = "idempotency"

    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]:
        last = ctx.audit.last_event_for(action.session_id, action.kind.value,
                                        action.plan_hash, action.attempt)
        if last is not None and last["event"] in (EXEC_STARTED, EXEC_OK):
            return Refusal(self.name, "duplicate",
                           f"This action already ran ({last['event']}).")
        return None


# ---------------------------------------------------------------------- gate

def _default_success(result: Any) -> bool:
    if result is None:
        return False
    success = getattr(result, "success", None)
    if isinstance(success, bool):
        return success
    if isinstance(result, bool):
        return result
    return True


class ActionGate:
    def __init__(self, policies: Sequence[Policy], audit: AuditLog,
                 gate_states: Optional[frozenset] = None):
        self.policies = list(policies)
        self.audit = audit
        self.gate_states = gate_states

    @classmethod
    def with_defaults(cls, *, kill_switch_env: str, audit: Optional[AuditLog] = None,
                      cap_usd: float = SpendCapPolicy.HARD_CAP_USD,
                      gate_states: Optional[frozenset] = None,
                      extra: Sequence[Policy] = ()) -> "ActionGate":
        audit = audit or AuditLog.from_env()
        defaults: List[Policy] = [
            KillSwitchPolicy(kill_switch_env),
            ApprovalPolicy(gate_states=gate_states),
            ConsentScopePolicy(),
            SpendCapPolicy(cap_usd),
            IdempotencyPolicy(),
        ]
        return cls(defaults + list(extra), audit, gate_states=gate_states)

    # ------------------------------------------------------------- approvals
    def record_approval(self, session, action: Action, *,
                        ttl_s: float = DEFAULT_APPROVAL_TTL_S,
                        max_uses: int = 1) -> Dict[str, Any]:
        """Bind the user's just-parsed YES to the exact plan they were shown.
        Persisted in session.slots so it survives restarts; pruned of dead
        entries on every write."""
        now = time.time()
        approvals = _approvals(session)
        approvals[:] = [a for a in approvals
                        if a.get("uses", 0) < a.get("max_uses", 1)
                        and now < a.get("expires_at", 0)]
        approval = {
            "kind": action.kind.value,
            "plan_hash": action.plan_hash,
            "scope": action.scope,
            "state": getattr(session, "fsm_state", None),
            "granted_at": now,
            "expires_at": now + ttl_s,
            "uses": 0,
            "max_uses": max_uses,
        }
        approvals.append(approval)
        return approval

    # ------------------------------------------------------------- execution
    async def execute(self, action: Action, session,
                      executor: Callable[[], Awaitable[T]],
                      success_of: Callable[[T], bool] = _default_success) -> GateOutcome[T]:
        ctx = GateContext(session=session, audit=self.audit, now=time.time())
        base = dict(session_id=action.session_id, workflow=action.workflow,
                    action_kind=action.kind.value, plan_hash=action.plan_hash,
                    attempt=action.attempt)

        for policy in self.policies:
            refusal = policy.check(action, ctx)
            if refusal is not None:
                self.audit.record(**base, event=GATE_DENY, policy=refusal.policy,
                                  code=refusal.code, summary=action.summary())
                logger.info("Gate refused %s (%s: %s)", action.kind.value,
                            refusal.code, refusal.detail)
                return GateOutcome(refusal=refusal)

        self.audit.record(**base, event=EXEC_STARTED, summary=action.summary())
        try:
            result = await executor()
        except EgressViolation as ev:
            # A boundary check stopped the payload mid-execution: surfaced as a
            # refusal (block-and-fail), never a crash or a silent redaction.
            self.audit.record(**base, event=EXEC_FAIL, code="egress",
                              policy="egress", summary=ev.reason)
            return GateOutcome(refusal=Refusal("egress", "egress_violation", str(ev)))
        except Exception as exc:
            self.audit.record(**base, event=EXEC_FAIL, code=type(exc).__name__)
            raise

        if success_of(result):
            self.audit.record(**base, event=EXEC_OK, summary=action.summary())
            approval = find_live_approval(session, action, ctx.now,
                                          gate_states=self.gate_states)
            if approval is not None:
                approval["uses"] = approval.get("uses", 0) + 1
        else:
            # Semantic failure (no answer, hand-off, mint refused): audited as
            # a failure and the approval is retained so a retry may proceed.
            self.audit.record(**base, event=EXEC_FAIL, code="unsuccessful")
        return GateOutcome(result=result)
