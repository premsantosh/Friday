# Harness Engineering — Technical Specification

**Status:** Implemented (H1–H5; see §12). Tests: `tests/test_harness.py` +
updated `tests/test_reservations.py`.
**Author:** Friday / Prem Santosh
**Last updated:** 2026-06-11
**Depends on:** `multi-turn-agent-spec.md` (sessions), `reservations-agent-spec.md` (first consumer)

---

## 1. Principle

**The LLM proposes; the harness disposes.** Anything deterministic, important, or
privacy-relevant is enforced by code at the boundary where it matters — not by prompt
discipline, not by call-site convention, not by the model behaving. The LLM's job is
reduced to what only an LLM can do (fuzzy language understanding, drafting prose); every
output it produces is a *proposal* that deterministic code validates, grounds, and gates
before anything acts on it.

A control that exists by convention is one missed call site away from not existing.
Today's reservation workflow proves it:

| Gap | Where it lives today |
|---|---|
| Kill switch not enforced at execution time | checked at routing (`workflow.py:192`) and in `BrowserChannel.commit` only; `EmailChannel.send`, `PhoneChannel.place_call`, and `SandboxBotChannel.run_bot` never check it — a flip while a session is `WAITING` doesn't stop the email/call |
| Approval not bound to the approved thing | "yes" commits whatever `CommitPlan` is in the in-memory `_pending` dict or rebuilt from slots; nothing proves committed == shown |
| Consent not bound to its scope | sandbox consent names the repo in prose only |
| FSM implicit | string states + if/elif in `resume()`; transitions undeclared, illegal ones unrepresented rather than impossible |
| Privacy rules unimplemented | spec §8 L1 (log redaction), L2 (session-DB PII purge), L5 (discovery query minimization) have **no code** — only careful prompt construction |
| Ambiguous values flow downstream | `date="next Friday"` stored verbatim; approved Tuesday, committed by a background tick Saturday = a different day |
| No idempotency / audit | a tick/restart race can double-fire `commit()`; no record of irreversible actions taken |

The harness is a new generic package, `core/harness/`, consumed first by reservations and
designed for every future workflow. **This pass migrates only reservations**; the simple
one-shot workflows are untouched.

### Decisions locked in (from design review)

| Question | Decision |
|---|---|
| Date handling | Normalize at slot-fill time with `python-dateutil`; confirmations display the resolved concrete date |
| Egress violation behavior | **Block and fail the step** (never redact-and-proceed silently) |
| Audit storage | SQLite, `~/.friday/audit.db`, alongside the session DB |
| Scope | Generic `core/harness/`; migrate reservations only |
| Confirmation parsing | Purely deterministic; no LLM anywhere on the approval path |

---

## 2. Package layout

```
core/harness/
├── __init__.py       # public surface re-exports
├── gate.py           # ActionGate, Action, ActionKind, policies, GateResult
├── audit.py          # AuditLog (SQLite, append-only events) + idempotency lookup
├── egress.py         # Sink declarations, guard(), scanners, log redaction filter, PII purge
├── fsm.py            # declarative StateMachine (states, events, transitions, gate states)
├── normalize.py      # SlotSpec + deterministic normalizers (date/time/party/email/phone)
├── extract.py        # LLMTask: schema + validators + grounding + fallback chain
└── confirm.py        # ConfirmationParser (yes/no/edit/unclear)

tests/test_harness.py # harness-only tests (no reservations imports)
```

Dependency rule: `core/harness/` imports nothing from `workflows/`. Workflows import the
harness, never the reverse. `python-dateutil` is added to `requirements.txt` (the only
new dependency).

---

## 3. `gate.py` — ActionGate

Every irreversible, outward-facing effect is an `Action` that can only execute by passing
through the gate. Channels stop self-policing and become pure executors.

### 3.1 Types

```python
class ActionKind(Enum):
    BOOK = "book"                          # browser form submit
    PLACE_CALL = "place_call"              # Bland dial
    SEND_EMAIL = "send_email"              # SMTP send
    MINT_CARD = "mint_card"                # Privacy.com single-use card
    RUN_UNTRUSTED_CODE = "run_untrusted_code"  # sandbox bot

@dataclass(frozen=True)
class Action:
    kind: ActionKind
    session_id: str
    workflow: str
    plan: Mapping[str, Any]       # JSON-serializable; exactly what the user saw
    scope: Optional[str] = None   # consent scope, e.g. "owner/repo" for sandbox
    amount_usd: Optional[float] = None  # for MINT_CARD

    @property
    def plan_hash(self) -> str:   # sha256 over canonical JSON (sort_keys, compact seps)
        ...
```

`plan_hash` is the spine of the design: the confirmation message is rendered *from*
`action.plan`, the user's "yes" is recorded *against* `plan_hash`, and the gate refuses
to execute any action whose hash doesn't match a live approval. "The thing committed is
the thing approved" becomes a checked invariant instead of a hope.

### 3.2 Approvals

```python
@dataclass
class Approval:
    plan_hash: str
    scope: Optional[str]
    granted_at: float
    consumed: bool = False
```

Approvals are persisted in `session.slots["_harness"]["approvals"]` (a reserved key the
framework already round-trips through SQLite), so they survive restarts — this replaces
the fragile in-memory `_pending` dict. The workflow records one only when
`ConfirmationParser` returns `YES` while the FSM is in a declared gate state:

```python
gate.record_approval(session, action)      # called by the workflow on a parsed YES
```

Approvals are **single-use** (consumed on successful execution) and carry a TTL
(default 24 h; watch/deferred-call flows re-gate anyway, so a stale yes can't fire days
later even if a code path forgets).

### 3.3 Policies and execution

```python
class Policy(Protocol):
    name: str
    def check(self, action: Action, ctx: GateContext) -> Optional[Refusal]: ...

@dataclass
class Refusal:
    policy: str
    code: str          # "kill_switch" | "no_approval" | "plan_mismatch" | "scope_mismatch"
                       # | "over_cap" | "duplicate"
    detail: str        # honest, loggable; the workflow maps codes to user-facing prose

class ActionGate:
    def __init__(self, policies: Sequence[Policy], audit: AuditLog): ...

    async def execute(self, action: Action, session: Session,
                      executor: Callable[[], Awaitable[T]]) -> GateOutcome[T]:
        # 1. run every policy at execution time (not routing time)
        # 2. on any Refusal: audit GATE_DENY, return GateOutcome(refusal=...)
        # 3. audit EXEC_STARTED; run executor; audit EXEC_OK / EXEC_FAIL
        # 4. consume the approval; return GateOutcome(result=...)
```

Shipped policies:

- **`KillSwitchPolicy(env_var)`** — re-reads the env var at the moment of execution.
  A mid-flight flip now blocks *every* action kind, everywhere, including ticks.
- **`ApprovalPolicy`** — a live, unconsumed, unexpired approval with a matching
  `plan_hash` must exist. Plan changed since the gate message? Hash differs → refused →
  workflow re-gates.
- **`ConsentScopePolicy`** — for `RUN_UNTRUSTED_CODE`, `approval.scope` must equal
  `action.scope` (the exact repo consented to).
- **`SpendCapPolicy(cap_usd=10.0)`** — for `MINT_CARD`, `action.amount_usd <= cap`.
  The cap stays in `PrivacyCardService` too — defense in depth, but the gate is the
  authoritative refusal.
- **`IdempotencyPolicy`** — refuses when the audit log already holds `EXEC_STARTED`/
  `EXEC_OK` for `(session_id, kind, plan_hash)`; for `EXEC_OK` the recorded outcome is
  returned instead of re-executing. A restart that replays a turn cannot double-book.

The default policy set applies to every gated workflow; workflows may add policies but
cannot remove the defaults (`ActionGate.with_defaults(extra=...)` is the only
constructor exposed to workflows).

---

## 4. `audit.py` — append-only audit log

SQLite at `~/.friday/audit.db` (`FRIDAY_AUDIT_DB` to override), file mode `0600`,
event-sourced — rows are never updated or deleted:

```sql
CREATE TABLE audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    session_id  TEXT NOT NULL,
    workflow    TEXT NOT NULL,
    action_kind TEXT NOT NULL,
    plan_hash   TEXT NOT NULL,
    event       TEXT NOT NULL,   -- GATE_DENY | EXEC_STARTED | EXEC_OK | EXEC_FAIL
    policy      TEXT,            -- refusing policy for GATE_DENY
    code        TEXT,            -- refusal code / error class
    summary     TEXT             -- redacted one-liner (business name, date, time only)
);
CREATE INDEX idx_audit_action ON audit_events(session_id, action_kind, plan_hash);
```

`summary` is built from an explicit whitelist of plan fields (`business_name`, `date`,
`time`, `party_size`) — never the whole plan, never card data, and it passes through the
egress scanner before insert (§5). The idempotency lookup is
`last_event_for(session_id, kind, plan_hash)`.

No retention limit for now (it's the safety record); a pruning utility can come later.

---

## 5. `egress.py` — boundary enforcement, redaction, purge

### 5.1 Sinks

Every place data leaves the process is a declared `Sink`. Two modes:

```python
class SinkMode(Enum):
    ALLOWLIST = "allowlist"   # only declared fields may appear; unknown keys refused
    SCAN = "scan"             # free-form payload, scanned for forbidden patterns

@dataclass(frozen=True)
class Sink:
    name: str                 # "llm" | "search" | "bland" | "sandbox" | "smtp" | ...
    mode: SinkMode
    allowed_fields: frozenset[str] = frozenset()

def guard(sink: Sink, payload: Union[str, Mapping]) -> Union[str, Mapping]:
    """Returns the payload unchanged, or raises EgressViolation. Block-and-fail:
    a violation aborts the step; nothing is silently redacted on the way out."""
```

Deterministic scanners (applied in both modes):

- **Card data:** 13–19 digit sequences (spaces/dashes tolerated) passing **Luhn**, and
  3–4 digit groups adjacent to `cvv`/`cvc`/`security code`.
- **Secrets:** values of any key matching `(api[_-]?key|token|password|secret|authorization)`,
  plus long high-entropy strings carrying known prefixes (`sk-`, `key-`, `Bearer `).

Reservation sink declarations (defined in the workflow, enforced by the harness):

| Sink | Mode | Allowed fields |
|---|---|---|
| `search` (Tavily/Yelp discovery) | ALLOWLIST | `business_name`, `location` — spec L5 becomes code |
| `llm.classify` (method classification) | ALLOWLIST | business evidence: `business`, `search_results` |
| `llm.extract` / `llm.draft` | SCAN | utterance/slots are the point; card/secret scan only |
| `bland` (call payload) | ALLOWLIST | `phone_number`, `task`, `business`, plus scan of the task text |
| `sandbox` (bot input) | ALLOWLIST | `business_name`, `date`, `time`, `party_size`, `url` — never credentials |
| `smtp` (email body) | SCAN | card/secret scan |
| `signal`, `calendar` | SCAN | user-owned destinations; scan only |

`EgressViolation` propagates to the workflow, which fails the step honestly
("I stopped that lookup, sir — it was about to include something it shouldn't").
Every violation is also written to the audit log.

### 5.2 Log redaction (spec L1)

`install_log_redaction()` attaches a `logging.Filter` to the root logger at startup
(called from `main.py`) that masks card-pattern matches, email addresses, and phone
numbers in every record's rendered message (`4242…4242 → ████`, `a@b.com → a***@***`,
last-4-preserved phones). Idempotent; covers all module loggers including
`conversations.log`.

### 5.3 PII purge (spec L2)

`purge_slots(session, pii_keys) -> None` blanks the listed keys in `session.slots`.
Hook: `ConversationalWorkflow` gains an overridable

```python
pii_slots: tuple[str, ...] = ()
def on_terminal(self, session: Session) -> None:   # default: purge_slots(session, self.pii_slots)
```

and `SessionManager` calls `on_terminal` once when a session enters a terminal status
(DONE/CANCELLED/EXPIRED), before the final save. This is the one small framework touch
in this design (additive, default no-op for existing workflows). Reservations declares
`pii_slots = ("guest_name", "phone", "email", "special_requests", "raw_request")` —
booking facts (business, date, time, confirmation number) are retained.

---

## 6. `fsm.py` — declarative state machine

```python
@dataclass(frozen=True)
class Machine:
    name: str
    states: frozenset[str]
    initial: str
    transitions: Mapping[tuple[str, str], str]   # (state, event) -> next state
    gate_states: frozenset[str]                  # only these may record approvals

    def fire(self, session: Session, event: str) -> str:
        """Validates (session.fsm_state, event), returns the next state.
        Unknown pair -> IllegalTransition (audited, surfaced as an honest error)."""
```

The workflow's `resume()`/`on_tick()` stop being if/elif ladders over raw strings:
handlers are registered per state (`@machine.handler("CONFIRM_BOOKING")`), the
dispatcher routes to them, and every state change goes through `fire()`. Dynamic
`collect_<slot>` states are replaced by a single `COLLECTING` state with the pending
slot name in `session.slots["_harness"]["pending_slot"]` — states are a closed set.

Reservation machine (same shape as spec §6, now declared):

```
states:  START, COLLECTING, CONFIRM_BOOKING, CONFIRM_EMAIL, CONFIRM_WAIT,
         CONFIRM_SANDBOX, CALLING, AWAITING_REPLY, WATCHING, DONE, CANCELLED
gates:   CONFIRM_BOOKING, CONFIRM_EMAIL, CONFIRM_WAIT, CONFIRM_SANDBOX
events:  NEED_SLOT, SLOT_FILLED, ESSENTIALS_READY, AVAILABLE, UNAVAILABLE,
         YES, NO, EDIT, UNCLEAR, CALL_PLACED, CALL_CONFIRMED, CALL_FAILED,
         EMAIL_REQUESTED, EMAIL_SENT, REPLY_CONFIRMED, REPLY_DECLINED,
         SLOT_OPENED, DEADLINE, GIVE_UP
```

`ApprovalPolicy` additionally checks that the approval was recorded while
`fsm_state ∈ gate_states` — approving from a non-gate state is structurally impossible.

The FSM module is fully generic: a future workflow defines its own `Machine` and gets
the same guarantees.

---

## 7. `normalize.py` — canonicalize at the boundary

Generic slot specification + deterministic normalizers:

```python
@dataclass(frozen=True)
class SlotSpec:
    name: str
    prompt: str                    # first ask
    reask: str                     # after a failed normalize ("I didn't catch a date…")
    normalize: Callable[[str, NormalizeCtx], Optional[Any]]
    required: bool = True

class NormalizeCtx:   # injected "now" + timezone, so tests are deterministic
    now: datetime
```

Normalizers (all pure functions, `dateutil`-backed, returning `None` on failure → the
slot-filling loop re-asks with `reask` instead of storing garbage):

- `normalize_date(text, ctx) -> "YYYY-MM-DD"` — handles `today/tonight/tomorrow`,
  `this weekend`, `(next|this) <weekday>`, `6/14`, `June 14`, full dateutil parse.
  Validates: not in the past, ≤ 366 days out. Relative dates are resolved **once, at
  slot-fill time**; the raw utterance is kept in `date_raw` for display context only.
- `normalize_time(text) -> "HH:MM"` (24 h) — `7pm`, `7:30 p.m.`, `19:30`, `noon`,
  `half past seven` (the LLM extractor already converts most phrasing; this is the
  validator of record).
- `normalize_party_size(text) -> int` in `[1, 100]`.
- `normalize_email`, `normalize_phone` (digits + `+`, 10–15 digits).
- Display helpers: `display_date("2026-06-19") -> "Friday, June 19"`,
  `display_time("19:00") -> "7:00 pm"`.

Consequences:

- The confirmation gate renders **resolved** values — the user approves
  "Friday, June 19 at 7:00 pm", a concrete fact with a stable hash, not "next Friday".
- `CalendarService._parse_date/_parse_time` are deleted; the calendar, email drafter,
  call plan, and watch deadlines all consume canonical values.
- `ESSENTIAL_SLOTS`/`SLOT_PROMPTS` in `models.py` become a `tuple[SlotSpec, ...]`, and
  the generic slot-filling loop (`fill_next(specs, slots, text) -> ask | done`) moves
  into the harness for reuse.

---

## 8. `extract.py` — structured LLM task harness

Generalizes the three bespoke validations in `llm.py` into one declarative shape:

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: type                                  # str | int | float | bool
    max_len: int = 300
    valid: Optional[Callable[[Any], bool]] = None
    grounded: bool = False    # value must literally appear in the evidence string
    reject_if: Optional[Callable[[Any], bool]] = None   # e.g. filler business names

@dataclass(frozen=True)
class LLMTask:
    name: str
    system_prompt: str
    fields: tuple[FieldSpec, ...]
    sink: Sink                                  # egress-guarded before the call
    fallback: Optional[Callable] = None         # deterministic path (regex/template)

@dataclass
class TaskResult:
    values: dict[str, Any]      # only fields that passed validation
    provenance: str             # "llm" | "fallback" | "none"
```

`run_task(llm, task, payload, evidence=None)`:

1. `guard(task.sink, payload)` — egress check before anything is sent.
2. One-shot JSON completion (the existing `ReservationLLM.complete_json` machinery
   moves here as the generic client).
3. Per-field validation: type, length, `valid`, `reject_if`, and **grounding**
   (`grounded=True` fields are dropped unless the value appears verbatim in the
   evidence — the existing anti-hallucination rule, now declarative).
4. Any failure ⇒ run `fallback` and tag provenance. An invalid LLM answer is
   indistinguishable from no LLM.

Reservations then *declares* its three tasks (slot extraction with the regex fallback,
method classification with the heuristic fallback and grounded `url`/`email`, email
drafting with the template fallback); `workflows/reservations/llm.py` shrinks to those
declarations plus prompts.

---

## 9. `confirm.py` — deterministic confirmation parsing

```python
class ConfirmDecision(Enum):
    YES = "yes"; NO = "no"; EDIT = "edit"; UNCLEAR = "unclear"

def parse_confirmation(text: str, *, editable: bool = False) -> ConfirmDecision
```

- Affirmative/negative keyword sets centralized (extracted from `_is_affirmative` plus
  the negative set currently inlined in `_handle_email_confirmation`).
- `editable=True` (email-draft gate): unrecognized input ⇒ `EDIT` (current re-draft
  behavior preserved).
- `editable=False` (booking/call/sandbox gates): unrecognized input ⇒ `UNCLEAR`, and the
  workflow **re-asks once** instead of cancelling. This deliberately changes today's
  behavior where any non-affirmative reply ("wait, what time was that?") silently
  cancels the booking; a second consecutive UNCLEAR cancels. Tests updated accordingly.
- No LLM on this path, ever.

---

## 10. Reservations refactor (consumer changes)

| File | Change |
|---|---|
| `workflow.py` | States/dispatch via the `Machine` (§6); `_pending`/`_pending_sandbox` dicts deleted — plan + approval live in slots; commit points call `gate.execute(...)` with executors; refusal codes mapped to the existing butler-voice messages; UNCLEAR re-ask added; confirmation messages rendered from `action.plan` with `display_date/time` |
| `channels/base.py` | Kill-switch check **removed** from `commit()` (gate owns policy); Playwright-missing and per-site hand-offs stay (operational concerns, not policy) |
| `channels/email.py` | `send()` reachable only via gate; SMTP body passes the `smtp` scan sink |
| `channels/phone.py` | `place_call()` reachable only via gate; Bland payload built through the `bland` allowlist sink |
| `channels/sandbox_bot.py` | `run_bot()` gated as `RUN_UNTRUSTED_CODE` with `scope=candidate.full_name`; bot input through the `sandbox` allowlist sink |
| `payment.py` | Mint gated as `MINT_CARD` with `amount_usd`; in-service cap kept as second layer |
| `discovery.py` | Search/Yelp queries through the `search` allowlist sink; classification via the declared `LLMTask` |
| `llm.py` | Shrinks to task declarations + prompts; generic client/validation move to `core/harness/extract.py` |
| `models.py` | `ESSENTIAL_SLOTS`/`SLOT_PROMPTS` → `tuple[SlotSpec, ...]` |
| `calendar.py` | `_parse_date/_parse_time` deleted; consumes canonical `date`/`time` |
| `base.py` (workflows) + `core/conversation/manager.py` | additive `pii_slots`/`on_terminal` hook (§5.3) |
| `main.py` | `install_log_redaction()` at startup |

Not gated (deliberately): availability checks and discovery reads (no side effects),
calendar/Signal writes (user-owned destinations, never fail the booking — scan-sink
only).

---

## 11. Testing

Existing `tests/test_reservations.py` stays green except the two deliberate behavior
changes (UNCLEAR re-ask; normalized dates in confirmation copy), which are updated in
the same commit that lands the change. New `tests/test_harness.py` covers, at minimum:

- **Gate:** kill-switch flipped between approval and execution blocks email/phone/
  sandbox (the live gap); plan mutated after approval → `plan_mismatch`; sandbox scope
  swap → `scope_mismatch`; replayed commit returns the recorded outcome (no double
  booking); `MINT_CARD` over $10 refused at the gate even if the service would mint.
- **Egress:** card number (Luhn) in an email body blocks the send; non-allowlisted
  field in a discovery query blocks the lookup; violations audited.
- **Redaction:** card/email/phone masked through a real logging handler.
- **Purge:** terminal session loses `pii_slots`, keeps booking facts.
- **FSM:** illegal transition raises; approval recorded outside a gate state is
  rejected by `ApprovalPolicy`.
- **Normalize:** date/time/party matrix incl. past-date rejection and re-ask loop;
  deterministic via injected `now`.
- **Extract:** ungrounded URL dropped; invalid JSON → fallback provenance.

---

## 12. Delivery order (each step lands wired into reservations, independently shippable)

1. **H1 — Gate + audit.** `gate.py`, `audit.py`; all five commit points gated;
   `_pending` dicts removed. *Biggest safety win; closes the kill-switch and
   double-fire gaps.*
2. **H2 — Egress + redaction + purge.** `egress.py`; sink declarations; log filter in
   `main.py`; `pii_slots` hook. *Spec §8 L1/L2/L5 become code.*
3. **H3 — FSM.** `fsm.py`; reservation machine declared; dispatch table replaces
   if/elif; gate-state check added to `ApprovalPolicy`.
4. **H4 — Normalize.** `normalize.py` + `dateutil`; SlotSpecs; calendar simplification;
   resolved-date confirmations.
5. **H5 — Extract + confirm.** `extract.py`, `confirm.py`; `llm.py` shrinks; UNCLEAR
   re-ask behavior.
