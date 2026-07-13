"""
Tests for core/harness — the deterministic enforcement layer (docs/harness-spec.md).

Gate tests run against the real ActionGate + policies with an in-memory audit
DB and a minimal stand-in session (the gate only touches .slots/.session_id/
.fsm_state).
"""

import time

import pytest

from core.harness import (
    Action,
    ActionGate,
    ActionKind,
    AuditLog,
    EXEC_OK,
    EgressViolation,
    RedactionFilter,
    Sink,
    SinkMode,
    find_live_approval,
    guard,
    hash_plan,
    purge_slots,
    redact_text,
)

# A Luhn-valid Visa test number — must never pass any egress sink.
CARD = "4111 1111 1111 1111"


class StubSession:
    def __init__(self):
        self.session_id = "s1"
        self.fsm_state = "confirm"
        self.slots = {}


def make_gate():
    return ActionGate.with_defaults(kill_switch_env="TEST_KILL_SWITCH",
                                    audit=AuditLog(":memory:"))


def book_action(session, plan=None):
    return Action(kind=ActionKind.BOOK, session_id=session.session_id,
                  workflow="test", plan=plan or {"business_name": "Lazy Bear",
                                                 "date": "friday", "time": "7pm"})


async def ok_executor():
    return True


# ------------------------------------------------------------------- hashing

def test_hash_plan_is_canonical():
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert hash_plan(a) == hash_plan(b)
    assert hash_plan(a) != hash_plan({**a, "b": 2})


# ---------------------------------------------------------------------- gate

@pytest.mark.asyncio
async def test_no_approval_refused():
    gate, session = make_gate(), StubSession()
    outcome = await gate.execute(book_action(session), session, ok_executor)
    assert not outcome.ok
    assert outcome.refusal.code == "no_approval"


@pytest.mark.asyncio
async def test_approved_action_executes_once_then_duplicates():
    gate, session = make_gate(), StubSession()
    action = book_action(session)
    gate.record_approval(session, action)

    first = await gate.execute(action, session, ok_executor)
    assert first.ok and first.result is True

    # Same action again: approval consumed AND audit shows EXEC_OK → refused.
    second = await gate.execute(action, session, ok_executor)
    assert not second.ok
    assert second.refusal.code in ("no_approval", "duplicate")


@pytest.mark.asyncio
async def test_plan_mutation_after_approval_is_refused():
    gate, session = make_gate(), StubSession()
    shown = book_action(session, plan={"business_name": "Lazy Bear", "time": "7pm"})
    gate.record_approval(session, shown)

    mutated = book_action(session, plan={"business_name": "Lazy Bear", "time": "9pm"})
    outcome = await gate.execute(mutated, session, ok_executor)
    assert not outcome.ok
    assert outcome.refusal.code == "no_approval"


@pytest.mark.asyncio
async def test_kill_switch_checked_at_execution_time(monkeypatch):
    gate, session = make_gate(), StubSession()
    action = book_action(session)
    gate.record_approval(session, action)          # approved while switch is off

    monkeypatch.setenv("TEST_KILL_SWITCH", "1")    # flipped mid-flight
    outcome = await gate.execute(action, session, ok_executor)
    assert not outcome.ok
    assert outcome.refusal.code == "kill_switch"


@pytest.mark.asyncio
async def test_consent_scope_binds_untrusted_code_to_the_repo():
    gate, session = make_gate(), StubSession()
    plan = {"business_name": "Lazy Bear"}
    consented = Action(kind=ActionKind.RUN_UNTRUSTED_CODE, session_id="s1",
                       workflow="test", plan=plan, scope="acme/ot-bot")
    gate.record_approval(session, consented)

    swapped = Action(kind=ActionKind.RUN_UNTRUSTED_CODE, session_id="s1",
                     workflow="test", plan=plan, scope="evil/other-bot")
    outcome = await gate.execute(swapped, session, ok_executor)
    assert not outcome.ok
    assert outcome.refusal.code == "scope_mismatch"


@pytest.mark.asyncio
async def test_mint_over_cap_refused_even_with_approval():
    gate, session = make_gate(), StubSession()
    action = Action(kind=ActionKind.MINT_CARD, session_id="s1", workflow="test",
                    plan={"purpose": "deposit", "amount_usd": 25.0}, amount_usd=25.0)
    gate.record_approval(session, action)
    outcome = await gate.execute(action, session, ok_executor)
    assert not outcome.ok
    assert outcome.refusal.code == "over_cap"


@pytest.mark.asyncio
async def test_expired_approval_is_dead():
    gate, session = make_gate(), StubSession()
    action = book_action(session)
    gate.record_approval(session, action, ttl_s=0.0)
    time.sleep(0.01)
    outcome = await gate.execute(action, session, ok_executor)
    assert not outcome.ok
    assert outcome.refusal.code == "no_approval"


@pytest.mark.asyncio
async def test_failed_execution_keeps_approval_for_retry():
    gate, session = make_gate(), StubSession()
    action = Action(kind=ActionKind.PLACE_CALL, session_id="s1", workflow="test",
                    plan={"business_name": "Lazy Bear"}, attempt=0)
    gate.record_approval(session, action, max_uses=3)

    async def no_answer():
        return None   # default success check: None → failure

    failed = await gate.execute(action, session, no_answer)
    assert failed.ok and failed.result is None      # gate allowed it; executor failed
    assert find_live_approval(session, action) is not None  # retry still possible

    # Retry as a new attempt (separate idempotency key) succeeds.
    retry = Action(kind=action.kind, session_id="s1", workflow="test",
                   plan=action.plan, attempt=1)

    async def answers():
        return "call_42"

    second = await gate.execute(retry, session, answers)
    assert second.ok and second.result == "call_42"


# ----------------------------------------------------------------------- fsm

from core.harness import IllegalTransition, Machine  # noqa: E402


def small_machine():
    return Machine(
        name="m", states=frozenset({"a", "gate", "b"}), initial="a",
        transitions={("a", "go"): "gate", ("gate", "done"): "b"},
        gate_states=frozenset({"gate"}),
    )


def test_machine_validates_declarations():
    with pytest.raises(ValueError):
        Machine(name="m", states=frozenset({"a"}), initial="nope", transitions={})
    with pytest.raises(ValueError):
        Machine(name="m", states=frozenset({"a"}), initial="a",
                transitions={("a", "e"): "ghost"})
    with pytest.raises(ValueError):
        Machine(name="m", states=frozenset({"a"}), initial="a", transitions={},
                gate_states=frozenset({"ghost"}))


def test_undeclared_transition_raises():
    m = small_machine()
    assert m.next("a", "go") == "gate"
    with pytest.raises(IllegalTransition):
        m.next("a", "done")
    with pytest.raises(IllegalTransition):
        m.next("b", "go")


@pytest.mark.asyncio
async def test_approval_outside_gate_state_is_refused():
    machine = small_machine()
    gate = ActionGate.with_defaults(kill_switch_env="TEST_KILL_SWITCH",
                                    audit=AuditLog(":memory:"),
                                    gate_states=machine.gate_states)
    session = StubSession()
    action = book_action(session)

    session.fsm_state = "a"                      # NOT a gate state
    gate.record_approval(session, action)
    outcome = await gate.execute(action, session, ok_executor)
    assert not outcome.ok and outcome.refusal.code == "no_approval"

    session.fsm_state = "gate"                   # proper gate state
    gate.record_approval(session, action)
    outcome = await gate.execute(action, session, ok_executor)
    assert outcome.ok


# -------------------------------------------------------------------- egress

SCAN_SINK = Sink("test_scan", SinkMode.SCAN)
ALLOW_SINK = Sink("test_allow", SinkMode.ALLOWLIST, frozenset({"business_name", "location"}))


def test_scan_blocks_card_numbers_but_not_phones():
    with pytest.raises(EgressViolation):
        guard(SCAN_SINK, f"please hold it with {CARD}, thanks")
    # Ten-digit phone numbers and ordinary text pass.
    guard(SCAN_SINK, "call me back at 415-555-1234, party of 4 at 7pm")


def test_scan_blocks_cvv_and_secret_shapes():
    with pytest.raises(EgressViolation):
        guard(SCAN_SINK, "the CVV is 123")
    with pytest.raises(EgressViolation):
        guard(SCAN_SINK, "header: Bearer abcdefghijklmnopqrstuv")
    with pytest.raises(EgressViolation):
        guard(SCAN_SINK, {"api_key": "x"})   # secret-named field


def test_allowlist_blocks_undeclared_fields():
    guard(ALLOW_SINK, {"business_name": "Lazy Bear", "location": "SF"})
    with pytest.raises(EgressViolation) as exc:
        guard(ALLOW_SINK, {"business_name": "Lazy Bear", "guest_phone": "+1999"})
    assert "guest_phone" in str(exc.value)


def test_allowlist_still_scans_values():
    with pytest.raises(EgressViolation):
        guard(ALLOW_SINK, {"business_name": f"Lazy Bear {CARD}"})


def test_phone_named_fields_skip_card_scan():
    # A declared phone field may hold a long Luhn-colliding number.
    sink = Sink("t", SinkMode.ALLOWLIST, frozenset({"phone_number"}))
    guard(sink, {"phone_number": "4111111111111111"})


@pytest.mark.asyncio
async def test_egress_violation_inside_gated_executor_becomes_refusal():
    gate, session = make_gate(), StubSession()
    action = book_action(session)
    gate.record_approval(session, action)

    async def leaky():
        guard(SCAN_SINK, f"body with {CARD}")

    outcome = await gate.execute(action, session, leaky)
    assert not outcome.ok
    assert outcome.refusal.code == "egress_violation"


# ----------------------------------------------------------------- redaction

def test_redact_text_masks_cards_emails_phones():
    out = redact_text(f"guest a@b.com, card {CARD}, phone +1 415 555 1234")
    assert "4111" not in out and "████CARD████" in out
    assert "a@b.com" not in out and "a***@***" in out
    assert "415 555" not in out and "1234" in out   # last four kept


def test_redaction_filter_on_a_real_handler():
    import io
    import logging as _logging

    stream = io.StringIO()
    handler = _logging.StreamHandler(stream)
    handler.addFilter(RedactionFilter())
    log = _logging.getLogger("harness.redaction.test")
    log.addHandler(handler)
    log.propagate = False
    try:
        log.warning("card seen: %s", CARD)
    finally:
        log.removeHandler(handler)
    assert "4111" not in stream.getvalue()
    assert "████CARD████" in stream.getvalue()


# --------------------------------------------------------------------- purge

def test_purge_slots_blanks_pii_keeps_facts():
    session = StubSession()
    session.slots = {"guest_name": "Prem", "phone": "+1999",
                     "business_name": "Lazy Bear", "date": "friday"}
    purge_slots(session, ("guest_name", "phone", "email"))
    assert session.slots["guest_name"] is None
    assert session.slots["phone"] is None
    assert session.slots["business_name"] == "Lazy Bear"


# ----------------------------------------------------------------- normalize

from datetime import datetime  # noqa: E402

from core.harness import (  # noqa: E402
    NormalizeCtx,
    display_date,
    display_time,
    normalize_date,
    normalize_party_size,
    normalize_phone,
    normalize_time,
)

# Thursday, 2026-06-11 09:00 — fixed clock for deterministic resolution.
CTX = NormalizeCtx(now=datetime(2026, 6, 11, 9, 0))


def test_normalize_date_matrix():
    assert normalize_date("today", CTX) == "2026-06-11"
    assert normalize_date("tonight", CTX) == "2026-06-11"
    assert normalize_date("tomorrow", CTX) == "2026-06-12"
    assert normalize_date("this weekend", CTX) == "2026-06-13"
    assert normalize_date("friday", CTX) == "2026-06-12"
    assert normalize_date("next friday", CTX) == "2026-06-12"
    assert normalize_date("thursday", CTX) == "2026-06-18"       # bare today-name → next week
    assert normalize_date("6/14", CTX) == "2026-06-14"
    assert normalize_date("6/1", CTX) == "2027-06-01"            # passed → next year
    assert normalize_date("June 14", CTX) == "2026-06-14"
    assert normalize_date("2026-06-20", CTX) == "2026-06-20"


def test_normalize_date_rejects_garbage_past_and_far_future():
    assert normalize_date("someday", CTX) is None
    assert normalize_date("", CTX) is None
    assert normalize_date("2026-06-01", CTX) is None             # explicit past date
    assert normalize_date("2028-01-01", CTX) is None             # > a year out


def test_normalize_time_matrix():
    assert normalize_time("7pm") == "19:00"
    assert normalize_time("7:30 P.M.") == "19:30"
    assert normalize_time("11am") == "11:00"
    assert normalize_time("19:30") == "19:30"
    assert normalize_time("noon") == "12:00"
    assert normalize_time("half past seven") == "19:30"          # evening rule
    assert normalize_time("half past seven in the morning") == "07:30"
    assert normalize_time("quarter to eight") == "19:45"
    assert normalize_time("seven o'clock") == "19:00"
    assert normalize_time("7") == "19:00"                        # bare hour → evening
    assert normalize_time("later") is None
    assert normalize_time("25:00") is None


def test_normalize_party_and_phone():
    assert normalize_party_size("for 4 people") == 4
    assert normalize_party_size("four of us") == 4
    assert normalize_party_size("0") is None
    assert normalize_party_size("400") is None
    assert normalize_phone("+1 (415) 555-1234") == "+14155551234"
    assert normalize_phone("call me") is None


def test_display_helpers_render_resolved_facts():
    assert display_date("2026-06-19") == "Friday, June 19"
    assert display_time("19:00") == "7:00 pm"
    # Unresolvable input passes through rather than crashing a message.
    assert display_date("the requested date") == "the requested date"
    assert display_time(None) == "None"


# ------------------------------------------------------------------- confirm

from core.harness import ConfirmDecision, parse_confirmation  # noqa: E402


def test_parse_confirmation_matrix():
    yes = ConfirmDecision.YES
    no = ConfirmDecision.NO
    assert parse_confirmation("yes") == yes
    assert parse_confirmation("Yes please!") == yes
    assert parse_confirmation("go ahead and book it") == yes
    assert parse_confirmation("sure, sounds good") == yes
    assert parse_confirmation("no") == no
    assert parse_confirmation("never mind") == no
    assert parse_confirmation("don't book it") == no       # negation beats "book it"
    assert parse_confirmation("not now") == no
    assert parse_confirmation("wait, what time was that?") == ConfirmDecision.UNCLEAR
    assert parse_confirmation("make it 7:30 instead",
                              editable=True) == ConfirmDecision.EDIT


def test_parse_confirmation_negation_wins_anywhere():
    """Regression: declined bookings used to parse as YES when the negation
    wasn't the first token — the embedded "book it" phrase matched first."""
    no = ConfirmDecision.NO
    assert parse_confirmation("No, don't book it.") == no      # comma-attached "No,"
    assert parse_confirmation("please don't book it") == no    # negation mid-sentence
    assert parse_confirmation("actually no, go ahead and cancel") == no
    assert parse_confirmation("do not send it") == no
    assert parse_confirmation("stop, don't do it") == no
    # Affirmative idioms containing "not" still approve.
    assert parse_confirmation("sure, why not") == ConfirmDecision.YES
    assert parse_confirmation("why not") == ConfirmDecision.YES


# ------------------------------------------------------------------- extract

from core.harness import FieldSpec, LLMTask, run_task  # noqa: E402


class FakeJsonClient:
    def __init__(self, response):
        self._response = response

    def complete_json(self, system, user, max_tokens=700):
        return self._response


TASK = LLMTask(
    name="t", system_prompt="extract", sink=SCAN_SINK,
    fields=(
        FieldSpec("method", str, required=True, valid=lambda m: m in {"a", "b"}),
        FieldSpec("url", str, grounded=True),
        FieldSpec("count", int, coerce=lambda v: int(v), valid=lambda n: n > 0),
    ),
)


def test_run_task_validates_and_grounds():
    client = FakeJsonClient({"method": "a", "url": "https://real.example",
                             "count": "3"})
    res = run_task(client, TASK, "payload", evidence="see https://real.example today")
    assert res.from_llm
    assert res.values == {"method": "a", "url": "https://real.example", "count": 3}

    # The same URL without supporting evidence is dropped; the rest survives.
    res = run_task(client, TASK, "payload", evidence="no links here")
    assert res.from_llm
    assert "url" not in res.values


def test_run_task_falls_back_when_required_field_fails():
    client = FakeJsonClient({"method": "zzz", "count": 3})   # invalid required field
    res = run_task(client, TASK, "payload", fallback=lambda: {"method": "a"})
    assert res.provenance == "fallback"
    assert res.values == {"method": "a"}

    res = run_task(FakeJsonClient(None), TASK, "payload")    # call failed, no fallback
    assert res.provenance == "none" and res.values is None

    res = run_task(None, TASK, "payload", fallback=lambda: {"method": "b"})
    assert res.provenance == "fallback"                      # no client at all


def test_run_task_guards_payload_before_calling():
    client = FakeJsonClient({"method": "a"})
    with pytest.raises(EgressViolation):
        run_task(client, TASK, f"the card is {CARD}")


@pytest.mark.asyncio
async def test_audit_records_the_execution_trail():
    audit = AuditLog(":memory:")
    gate = ActionGate.with_defaults(kill_switch_env="TEST_KILL_SWITCH", audit=audit)
    session = StubSession()
    action = book_action(session)
    gate.record_approval(session, action)
    await gate.execute(action, session, ok_executor)

    last = audit.last_event_for("s1", "book", action.plan_hash)
    assert last is not None and last["event"] == EXEC_OK
    assert "Lazy Bear" in (last["summary"] or "")
