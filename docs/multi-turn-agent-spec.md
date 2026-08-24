# Friday Multi-Turn Agent Framework — Technical Specification

**Status:** Draft for review (no integration yet)
**Author:** Friday / Prem Santosh
**Last updated:** 2026-06-01

---

## 1. Summary

Friday today is **single-turn**: `VoiceAssistant.process_input()` matches one workflow,
returns one `WorkflowResult`, speaks it, and forgets everything
(`core/assistant.py:115`). This spec adds a **general-purpose multi-turn capability** that
any workflow can opt into — not just reservations. It is built from two complementary,
independently useful layers:

| Layer | Purpose | Lifetime | Backing |
|---|---|---|---|
| **A — Conversational Context Register** | Carry the last interaction forward so follow-ups resolve ("turn it down" → the thing we just discussed). Improves routing for **every** workflow. | ~3 turns, auto-expiring | [`turnstile-ctx`](https://github.com/premsantosh/turnstile-ctx) (stdlib, optional JSON) |
| **B — Task Session Framework** | Structured, possibly long-running tasks: slot-filling, confirmation gates, "wait and act later". Opt-in per workflow. | Minutes → days, durable | New `SessionStore` (SQLite) |

These are orthogonal. Layer A makes one-shot commands smarter. Layer B makes whole tasks
possible. Reservations is the **first consumer** of both, but the framework is designed so
HVAC scheduling, multi-step troubleshooting, ordering, etc. reuse it unchanged.

---

## 2. Why two layers (and why not just one)

`turnstile-ctx` is a **context register**, not a task engine. Per its README it stores the
last route's `active_domain / active_device / last_action / parameters`, enriches the next
utterance with a readable context prefix, and **expires after a configurable number of turns
(default 3)**. That is exactly right for follow-up disambiguation and wrong for a reservation
that takes six turns, pauses, and may complete days later.

So:
- **Layer A = turnstile-ctx, used as-is** for short-term routing continuity.
- **Layer B = a new durable session layer** for structured tasks.

When a Layer B task is active, we keep Layer A aligned to it (set the register's active domain
to the session's workflow) so the router naturally biases toward the in-progress task, but the
**session's slots are the source of truth**, not the 3-turn register.

---

## 3. Layer A — Conversational Context Register (turnstile-ctx)

### 3.1 Adapter

A thin wrapper isolates Friday from the dependency and applies Friday's config/ephemeral rules.

```
core/conversation/context.py
    class ConversationContext:
        def __init__(self, persist_path: str | None, expiry_turns: int = 3): ...
        def enrich(self, text: str) -> str          # → register.enrich(text).enriched_utterance
        def update(self, route_result, text) -> None # → register.update(result, text)
        def clear(self) -> None
```

Internally holds a `turnstile.ContextRegister(persistence_path=…)`. turnstile guarantees
`enrich`/`update` never raise, so this is safe to call unconditionally.

### 3.2 Integration into the assistant pipeline

Two additive calls in `process_input` (full revised pipeline in §6):
1. **Before routing:** `text = self.context.enrich(raw_text)` — the router now
   sees `"[context: domain=hvac, device=living room] turn it down"`.
2. **After a successful route:** `self.context.update(route, raw_text)` so the next turn
   inherits this domain/device/action.

This alone fixes follow-up commands across all existing workflows (Hue, HVAC, Shelly) with no
per-workflow code.

### 3.3 Lifecycle & ephemeral mode
- Persist register JSON at `~/.friday/context_register.json` in normal mode.
- In **ephemeral mode** (`LLMConfig.ephemeral`, used by `--chat`/`--test`), pass
  `persist_path=None` so test runs don't leak context — mirrors how the intent cache is
  disabled today (`main.py:160`).

---

## 4. Layer B — Task Session Framework (new)

### 4.1 Core data model

```python
# core/conversation/session.py

class SessionStatus(Enum):
    ACTIVE              # in live dialogue with the user right now
    AWAITING_CONFIRMATION  # waiting for an explicit yes/no before a commit
    WAITING             # detached; a background task will advance it (e.g. wait-and-book)
    DONE
    CANCELLED
    EXPIRED

@dataclass
class Session:
    session_id: str
    user_id: str                 # "default" until Friday is multi-user
    workflow_name: str           # owning ConversationalWorkflow
    fsm_state: str               # state name owned & interpreted by the workflow
    slots: dict[str, Any]        # collected, validated user data (source of truth)
    scratch: dict[str, Any]      # workflow-private working data (candidates, page handles…)
    status: SessionStatus
    created_at: float
    updated_at: float
    expires_at: float            # absolute; sweeper closes stale sessions
```

`fsm_state` and `slots` are **owned by the workflow** — the framework never interprets them.
That is what keeps the framework generic.

### 4.2 Turn results — how a workflow drives the conversation

Existing single-turn workflows return `WorkflowResult` (unchanged). Conversational workflows
return a `TurnResult` that adds a **control signal** telling the manager what to do with the
session:

```python
class TurnControl(Enum):
    CONTINUE             # stay ACTIVE; `message` asks the user for the next slot
    AWAIT_CONFIRMATION   # → AWAITING_CONFIRMATION; `message` is the summary to approve
    BACKGROUND           # → WAITING; detach, a background task will advance it
    COMPLETE             # → DONE; `message` is the final result
    CANCEL               # → CANCELLED

@dataclass
class TurnResult(WorkflowResult):       # extends existing dataclass
    control: TurnControl = TurnControl.COMPLETE
    slots_update: dict | None = None    # merged into session.slots by the manager
    next_state: str | None = None       # new fsm_state
    wake_at: float | None = None        # for BACKGROUND: when to next tick this session
```

The manager applies `slots_update`/`next_state`, transitions status per `control`, persists,
and returns `message` to be spoken. Single-turn `WorkflowResult` keeps working because
`TurnControl.COMPLETE` is the default.

### 4.3 The `ConversationalWorkflow` interface

```python
# workflows/base.py (added alongside Workflow)

class InterruptPolicy(Enum):
    STICKY            # default: all input goes to the session until done/cancelled
    ALLOW_READONLY    # let read-only one-off queries (time, weather) answer, then resume

class ConversationalWorkflow(Workflow):
    interrupt_policy: InterruptPolicy = InterruptPolicy.STICKY
    session_timeout_s: int = 600       # workflows override (reservations: much longer)

    async def start(self, intent: str, entities: dict, session: Session) -> TurnResult:
        """First turn: seed slots from the utterance, return the next prompt/confirmation."""

    async def resume(self, text: str, session: Session) -> TurnResult:
        """A subsequent user turn while this session is ACTIVE/AWAITING_CONFIRMATION."""

    async def on_tick(self, session: Session) -> TurnResult | None:
        """Optional: called by the background runner for WAITING sessions (poll availability,
        receive an async call outcome). Return a TurnResult to advance, or None to keep waiting."""
```

Backward compatible: a `ConversationalWorkflow` still satisfies the `Workflow` ABC. `execute()`
on the base can default to "call `start()` with a fresh session" so the existing
router pipeline can launch it.

### 4.4 Storage

```
core/conversation/store.py
    class SessionStore(ABC): get / get_active_for_user / list_waiting / save / delete
    class SqliteSessionStore(SessionStore)     # ~/.friday/sessions.db — durable, survives restart
    class InMemorySessionStore(SessionStore)   # ephemeral mode + tests
```

SQLite is required so long-running sessions (wait-and-book over days, a call placed an hour
ago) survive a Friday restart. `slots`/`scratch` are JSON columns. Note: live resources like a
Playwright page handle can't be serialized — they live in `scratch` only while the process is
up; on restart the workflow's `on_tick`/`resume` must be able to rebuild them from durable
slots.

### 4.5 SessionManager — lifecycle & routing

```
core/conversation/manager.py
    class SessionManager:
        has_active(user) -> bool                 # ACTIVE or AWAITING_CONFIRMATION only
        get_active(user) -> Session | None       # WAITING sessions are NOT "active dialogue"
        async open(workflow, intent, entities, user) -> TurnResult
        async handle(user, text) -> TurnResult    # route a turn to the active session's workflow
        cancel(user, reason) -> None
        sweep_expired() -> list[Session]          # called by the background runner
```

Multiple **WAITING** sessions can coexist (two pending reservations), but at most one
**ACTIVE** dialogue session per user — `get_active` deliberately ignores WAITING ones so a
backgrounded task never hijacks the next command.

### 4.6 Background runner (long-running work)

`core/conversation/background.py :: BackgroundTaskRunner` — one shared loop modeled on the
existing coffee-machine monitor (`main.py:351`). On each interval it:
1. `sweep_expired()` → close stale sessions.
2. For each `WAITING` session whose `wake_at` has passed → `workflow.on_tick(session)`;
   apply the returned `TurnResult`.
3. When a tick yields `COMPLETE` or `AWAIT_CONFIRMATION`, **notify the user** via the existing
   `assistant.speak` callback ("Sir, the table you wanted at Lazy Bear just opened — shall I
   confirm it?").

Reuses the exact `start_monitor(speak_callback)` pattern already wired in `main.py`, so no new
threading model.

---

## 5. Interrupt / barge-in handling (generic)

A real concern the moment dialogue spans turns. Policy, owned by the framework:

- **Global escape commands always win.** "cancel", "stop", "never mind", "forget it" end the
  active session regardless of workflow. Defined centrally.
- **Per-workflow `interrupt_policy`:**
  - `STICKY` (default): every other utterance is fed to the session's `resume()`.
  - `ALLOW_READONLY`: if the utterance clearly matches a workflow marked **read-only/idempotent**
    (time, weather), answer it as a one-off and then re-issue the session's pending prompt; the
    session stays `ACTIVE`. (Requires a `read_only: bool` flag on workflows.)
- **Timeout:** `session_timeout_s` per workflow; the background sweeper expires idle sessions so
  Friday never gets "stuck" mid-task.

For v1 I recommend shipping `STICKY` + global escapes + timeout (simple, predictable) and
adding `ALLOW_READONLY` once the basics are proven.

---

## 6. Revised `process_input` pipeline

```python
async def process_input(self, raw_text: str) -> str:
    # Layer A: enrich for routing/continuity
    text = self.context.enrich(raw_text)

    # Global escape always honored
    if self.sessions.has_active(user) and is_global_escape(raw_text):
        self.sessions.cancel(user, "user aborted")
        return "Cancelled, sir."

    # Layer B: an active dialogue session takes the turn
    if self.sessions.has_active(user):
        result = await self.sessions.handle(user, raw_text)   # raw text = slot content
        self.context.update_for_session(self.sessions.get_active(user))
        return result.message

    # ----- existing single-turn pipeline, now on enriched `text` -----
    # (historical sketch: the keyword/pattern fast-path shown here was later
    # removed; routing is intent cache -> agent engine / Claude router)
    ... intent cache ... Claude router ...                    # unchanged

    # When the selected workflow is conversational and isn't done in one shot:
    if isinstance(wf, ConversationalWorkflow):
        turn = await self.sessions.open(wf, intent, entities, user)
        self.context.update(route, raw_text)
        return turn.message     # if it returned CONTINUE, the session is now ACTIVE

    # else: legacy single-turn path, plus context.update(route, raw_text)
```

**Net change to existing code:** one new field (`self.context`, `self.sessions`), a guarded
block at the top of `process_input`, and `context.update(...)` after successful routes. When no
session is active and no conversational workflow is triggered, behaviour is **byte-for-byte the
same as today**.

---

## 7. Configuration

```python
# config/settings.py
@dataclass
class ConversationConfig:
    # Layer A
    context_enabled: bool = True
    context_persist_path: str = "~/.friday/context_register.json"
    context_expiry_turns: int = 3
    # Layer B
    sessions_enabled: bool = True
    session_store_path: str = "~/.friday/sessions.db"
    default_session_timeout_s: int = 600
    background_tick_seconds: int = 30
```
Added to `AssistantConfig`. In ephemeral mode: `context_persist_path=None` and an
`InMemorySessionStore`, mirroring intent-cache handling.

---

## 8. File layout

```
core/conversation/
├── __init__.py
├── context.py        # Layer A: ConversationContext (wraps turnstile-ctx)
├── session.py        # Session, SessionStatus, TurnControl, TurnResult
├── store.py          # SessionStore ABC + SqliteSessionStore + InMemorySessionStore
├── manager.py        # SessionManager (lifecycle, routing, escapes, expiry)
└── background.py     # BackgroundTaskRunner (ticks WAITING sessions, notifications)

workflows/base.py     # + ConversationalWorkflow, InterruptPolicy, read_only flag
core/assistant.py     # + self.context / self.sessions, revised process_input
config/settings.py    # + ConversationConfig
requirements.txt      # + turnstile-ctx @ git+https://github.com/premsantosh/turnstile-ctx.git@<pinned-sha>
main.py               # wire ConversationContext + SessionManager + start BackgroundTaskRunner
```

`turnstile-ctx` is stdlib-only (Python 3.10+, zero required deps), so it adds no transitive
weight. Pin a commit SHA.

---

## 9. Reuse examples (proof it isn't reservation-specific)

- **HVAC schedule builder** — `ConversationalWorkflow` collecting days/times/temps over turns,
  confirmation gate before applying.
- **Multi-step troubleshooting** — branch on user answers (`fsm_state`), no slots persisted long.
- **Ordering / shopping** — cart built across turns, `AWAIT_CONFIRMATION` before checkout,
  Privacy.com card reused from the reservations work.
- **Any "do X later when Y" task** — `BACKGROUND` + `on_tick`, e.g. "remind/act when the price
  drops".

The reservation workflow is just the first, most demanding consumer (uses all of: slot-filling,
confirmation gate, background wait, async call results).

---

## 10. Phased delivery

1. **MT1 — Layer A.** ✅ **Done.** `ConversationContext` (`core/conversation/context.py`) +
   turnstile-ctx pinned in `requirements.txt` + enrich-before-router / update-after-route in
   `process_input`. Enriched text feeds the LLM router only; cache matching stays on raw
   text so an injected context prefix can't trigger false cache hits.
2. **MT2 — Layer B core.** ✅ **Done.** `Session`/`TurnControl`/`TurnResult` (`session.py`),
   `SessionStore` + `SqliteSessionStore` + `InMemorySessionStore` (`store.py`), `SessionManager`
   (`manager.py`), the `ConversationalWorkflow` ABC (`workflows/base.py`), and the active-session
   branch in `process_input`. Validated by `tests/test_conversation.py` (slot-fill → confirm →
   complete, escape/cancel, expiry, SQLite round-trip).
3. **MT3 — Background runner.** ✅ **Done.** `BackgroundTaskRunner` (`background.py`) on a daemon
   thread (coffee-monitor pattern): `sweep_expired()` + `tick_waiting()` driving `on_tick`, with
   notification via the assistant's `speak` callback; started in `VoiceAssistant.run()`, stopped
   in `stop()`. Tested incl. an end-to-end thread test.
4. **MT4 — Interrupt polish.** ⏳ Not started. `ALLOW_READONLY` interrupts + `read_only` flags.

The Reservations agent (separate spec) starts after **MT2** (needs slot-filling + confirmation),
and consumes **MT3** for wait-and-book / async call/email outcomes — both now available.

---

## 11. Resolved decisions

1. **Multi-user** — single user for v1 (`user_id="default"`). Per-speaker sessions deferred.
2. **turnstile-ctx** — pinned by **git SHA** in `requirements.txt` (not vendored).
3. **Concurrency** — **one ACTIVE dialogue** at a time; many WAITING sessions may coexist.
```
