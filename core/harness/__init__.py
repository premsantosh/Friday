"""
Friday's harness layer: deterministic enforcement around LLM-assisted workflows.

The LLM proposes; the harness disposes. See docs/harness-spec.md.
"""

from .audit import EXEC_FAIL, EXEC_OK, EXEC_STARTED, GATE_DENY, AuditLog
from .egress import (
    EgressViolation,
    RedactionFilter,
    Sink,
    SinkMode,
    guard,
    install_log_redaction,
    purge_slots,
    redact_text,
)
from .confirm import ConfirmDecision, parse_confirmation
from .extract import FieldSpec, JsonLLMClient, LLMTask, TaskResult, parse_json_object, run_task
from .fsm import IllegalTransition, Machine
from .normalize import (
    NormalizeCtx,
    SlotSpec,
    display_date,
    display_time,
    normalize_date,
    normalize_email,
    normalize_party_size,
    normalize_phone,
    normalize_text,
    normalize_time,
)
from .gate import (
    Action,
    ActionGate,
    ActionKind,
    ApprovalPolicy,
    ConsentScopePolicy,
    GateOutcome,
    IdempotencyPolicy,
    KillSwitchPolicy,
    Policy,
    Refusal,
    SpendCapPolicy,
    find_live_approval,
    hash_plan,
)

__all__ = [
    "Action", "ActionGate", "ActionKind", "ApprovalPolicy", "ConsentScopePolicy",
    "GateOutcome", "IdempotencyPolicy", "KillSwitchPolicy", "Policy", "Refusal",
    "SpendCapPolicy", "find_live_approval", "hash_plan",
    "AuditLog", "GATE_DENY", "EXEC_STARTED", "EXEC_OK", "EXEC_FAIL",
    "EgressViolation", "RedactionFilter", "Sink", "SinkMode", "guard",
    "install_log_redaction", "purge_slots", "redact_text",
    "IllegalTransition", "Machine",
    "NormalizeCtx", "SlotSpec", "display_date", "display_time",
    "normalize_date", "normalize_email", "normalize_party_size",
    "normalize_phone", "normalize_text", "normalize_time",
    "ConfirmDecision", "parse_confirmation",
    "FieldSpec", "JsonLLMClient", "LLMTask", "TaskResult",
    "parse_json_object", "run_task",
]
