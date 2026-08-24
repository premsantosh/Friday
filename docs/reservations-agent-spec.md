# Reservations Agent — Technical Specification

**Status:** Draft for review (no integration yet)
**Author:** Friday / Prem Santosh
**Last updated:** 2026-06-01

---

## 1. Summary

A new Friday capability that takes a natural-language reservation request ("book me a
table for 2 at Lazy Bear next Friday at 7pm"), looks the business up online, figures out
*how* that business accepts reservations, and then completes the booking through the
appropriate channel — an online site driven by browser automation, a phone call placed by
an AI voice agent, or (as a fallback) a sandboxed third-party automation bot. When a
reservation requires a credit card, the agent mints a single-use Privacy.com virtual card
**hard-capped at $10** — always single-use, never more than $10, to bound the damage if the
card number ever leaks. If an establishment requires a hold/charge above $10, the agent
**aborts that booking and tells the user** rather than minting a larger card. When a slot
isn't yet available, the agent can wait and book the moment
it opens.

The agent is **human-in-the-loop**: it does all research and preparation autonomously, but
pauses for the user's explicit approval before any irreversible, outward-facing action
(submitting a booking, placing a call, charging a card).

### Decisions locked in (from spec review)

| Question | Decision |
|---|---|
| Autonomy level | **Confirm before each commit** — research is autonomous; final booking / call / card charge requires explicit user approval. |
| Phone calls | **Bland.ai** places the autonomous call from an agent-generated, user-approved call plan. Google Voice is *not* viable (no call API). Fallback: dial-and-bridge the user in. |
| OpenTable / Resy / Yelp | **Browser automation (Playwright)** against the real sites with the user's logged-in session — these platforms have no public consumer booking API. **Yelp** additionally used for discovery via its Fusion API. |
| No-API sites | **Prefer our own Playwright engine; fall back to a sandboxed (Docker) third-party GitHub bot.** Docker is available locally, so the sandbox fallback is enabled. |
| Calendar | On a confirmed booking, **create a Google Calendar event** with all details (Friday already has Google Calendar tools). |
| Async notifications | When Friday isn't actively listening, send a **Telegram** message; speak it when it is. |
| Accounts | User holds a **Privacy.com API key** and **OpenTable / Resy / Yelp** logins. |
| Multi-user / concurrency | Single user, **one active dialogue** at a time (many backgrounded). |

### Scope

- **Appointment-based businesses**, not just restaurants: hair salons, massage/spa,
  and similar bookable services, in addition to dining. Discovery heuristics and channels
  cover restaurant platforms (OpenTable, Resy, Yelp) **and** appointment platforms
  (e.g. Vagaro, Booksy, Square Appointments, Mindbody, Fresha) via the generic web engine.
- **Hotels are out of scope** for now — they need substantially more information (rooms,
  occupancy, rates, multi-night) and a different data model.

---

## 2. Goals & non-goals

### Goals
- Accept a reservation request in conversation; collect **basic details up front**, ask for
  the rest **only when a channel actually needs them**.
- Discover the business and **classify its reservation method** automatically.
- Book via OpenTable/Resy (browser), generic web forms (our Playwright engine), phone (AI
  voice), or sandboxed GitHub bot — in that order of preference.
- Use a **single-use Privacy.com card, hard-capped at $10**, whenever a card is required.
  Never mint above $10; if the establishment needs more, abort and tell the user.
- **Wait-and-book**: if a slot isn't open, poll/retry until it is (or a deadline passes),
  then notify the user.
- **Off-hours call deferral**: if a booking must be done by phone and the business is closed
  now, remember it and place the (already-approved) call when the business next opens.
- **Handle every call outcome**: confirmed, no availability, "email us a request", callback
  needed, or no-answer — each routed to a sensible next step (see §5.4).
- **Email channel**: when a business asks to be emailed (or only lists email), draft a
  reservation-request email, get the user's approval, send it, and watch for the reply.
- Never take an irreversible action without explicit user confirmation.

### Non-goals (v1)
- Modifying/cancelling existing reservations (read-only confirmation tracking only).
- Loyalty/points optimization, multi-restaurant comparison shopping.
- Running untrusted code outside a sandbox.
- Any card above $10 — the cap is hard; bookings needing a larger hold/charge are aborted
  (handed back to the user), never auto-escalated.

---

## 3. How it fits into Friday

Reservations are inherently **multi-turn** (gather details over several exchanges) and
**long-running** (calls and wait-and-book can take minutes to days). Friday today is
single-turn (`core/assistant.py:115`), so these capabilities are **not** built inside the
reservations workflow — they are provided by a **general-purpose multi-turn agent framework**
specced separately in **[`multi-turn-agent-spec.md`](./multi-turn-agent-spec.md)**. The
reservations agent is simply the **first consumer** of that framework.

### 3.1 Dependency on the multi-turn framework

The reservations workflow is implemented as a **`ConversationalWorkflow`** (defined by the
multi-turn spec), which gives it for free:

- **Layer A — context register (turnstile-ctx):** follow-up continuity ("make it 7:30 instead"
  resolves against the in-progress booking).
- **Layer B — task sessions:** a durable `Session` (SQLite) holds partial reservation state
  across turns. The reservation **state machine (§6) is the session's `fsm_state`**; the
  collected details are the session's `slots`. The agent drives the dialogue by returning
  `TurnResult`s with control signals (`CONTINUE` to ask for a slot, `AWAIT_CONFIRMATION` for the
  HITL gate, `BACKGROUND` for wait-and-book, `COMPLETE`/`CANCEL`).
- **Background runner:** wait-and-book and async phone-call outcomes are handled by the
  framework's `BackgroundTaskRunner` driving the workflow's `on_tick(session)`, with spoken
  notifications via the existing `assistant.speak` callback.

So the **only changes to `core/assistant.py`, the session store, and the background monitor are
owned by the multi-turn spec** — the reservations agent adds no bespoke session or task
infrastructure of its own.

### 3.2 Reused as-is
- `Workflow` / `WorkflowManager` ABC (`workflows/base.py`) — the agent is a (conversational) workflow.
- Conditional registration on env vars in `main.py:create_workflow_manager`.
- Tavily search (`search/provider.py`) for business discovery.
- Claude via the existing LLM provider for classification, slot extraction, and call-script
  generation.
- Env-var secret convention.

---

## 4. Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │            ReservationWorkflow               │
                        │  (Workflow subclass; entry point + routing)  │
                        └───────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────────┐
        │                               │                                   │
┌───────▼─────────────────┐ ┌───────────▼─────────────┐  ┌──────────────────▼──────────┐
│ Multi-turn framework    │ │ BusinessDiscovery       │  │ Multi-turn framework        │
│ SessionManager + Session│ │  - Tavily search        │  │ BackgroundTaskRunner        │
│ (slot-filling FSM +     │ │  - LLM method classifier│  │  → calls workflow.on_tick   │
│  HITL via TurnResult)   │ │  → ChannelDecision      │  │  (wait-and-book, async      │
│  [see multi-turn spec]  │ │                         │  │   call results, notify)     │
└───────┬─────────────────┘ └────────────┬────────────┘  └─────────────────────────────┘
        │                                │
        │                    ┌───────────▼─────────────────────────────────┐
        │                    │       ChannelRouter (strategy select)        │
        │                    └───────────┬─────────────────────────────────┘
        │                                │
        │   Channels (ReservationChannel impls), in preference order:
        │     OpenTable · Resy · Yelp   →  GenericWeb  →  Phone (Bland.ai)  →  SandboxBot
        │     (Playwright + logged-in      (our LLM-driven   (user-approved      (Docker-isolated
        │      session)                     form engine)      call plan)          GitHub bot)
        │
   ┌────▼─────────────────┐   ┌──────────────────────┐   ┌────────────────────────┐
   │ PrivacyCardService    │   │ CalendarService       │   │ TelegramNotifier       │
   │ (single-use, $10 def.)│   │ (Google Cal event on  │   │ (async updates when    │
   │ ← any requires_card   │   │  CONFIRMED)           │   │  Friday isn't listening)│
   └───────────────────────┘   └──────────────────────┘   └────────────────────────┘
```

### 4.1 Component responsibilities

**`ReservationWorkflow`** (`workflows/reservations/workflow.py`) — a **`ConversationalWorkflow`**
(from the multi-turn framework)
- Trigger: `WorkflowTrigger.examples` steer the LLM router and the agent's tool
  descriptions (the keyword/pattern fast-path was removed; routing is LLM-only).
- `start(intent, entities, session)`: extracts initial slots from the utterance into
  `session.slots`, kicks off discovery, returns the first `TurnResult`.
- `resume(text, session)`: drives the §6 state machine — slot-filling, then the confirmation
  reply. Returns `CONTINUE` while collecting, `AWAIT_CONFIRMATION` to present the summary,
  `COMPLETE`/`CANCEL` at the end.
- `on_tick(session)`: advances `WAITING` sessions (poll availability for wait-and-book, pick up
  an async phone-call result).
- Sets `session_timeout_s` high (reservations span far longer than a typical command) and uses
  the framework's confirmation/escape handling rather than rolling its own.

**Session lifecycle, slot-filling, the HITL confirmation gate, persistence, and timeouts are all
provided by the multi-turn framework's `SessionManager` / `Session` / `TurnResult`** — see
[`multi-turn-agent-spec.md`](./multi-turn-agent-spec.md) §4. The reservation FSM (§6) is encoded
in `session.fsm_state`; collected details live in `session.slots`.

**`BusinessDiscovery`** (`workflows/reservations/discovery.py`)
- **Yelp Fusion API** (business match/search + business details) as the primary structured
  source: resolves the business, its phone number, address, categories, any reservation/booking
  link, and **opening hours + timezone** (`hours`, `is_open_now`, `special_hours`). Falls back
  to **Tavily** web search for the official site / OpenTable / Resy / appointment-platform pages
  (and for hours if Yelp lacks them). Hours/timezone are stored in `session.slots` and drive
  off-hours call deferral (§5.4).
- LLM **method classifier** consumes the structured + search results and returns a
  `ChannelDecision`: `{ method: opentable|resy|yelp|generic_web|phone|email|unknown, url, phone,
  email, confidence, requires_card_hint, notes }`. It recognizes restaurant platforms (OpenTable,
  Resy, Yelp) and appointment platforms (Vagaro, Booksy, Square Appointments, Mindbody,
  Fresha → `generic_web`).
- "Online booking not mentioned" or low confidence ⇒ defaults to **phone** (per the autonomy
  rules), or `generic_web` if a booking form/link was found.

**`ChannelRouter`** — picks the concrete channel from the decision, applying the preference
order: OpenTable/Resy/Yelp browser → GenericWeb → Phone (Bland.ai) → Email → SandboxBot. The
**Email** channel is also entered *mid-flow* when a phone call ends in "email us a request".

**`ReservationChannel` (interface)** — every channel implements:
```python
class ReservationChannel(ABC):
    name: str
    async def required_slots(self, reservation) -> list[Slot]: ...
    async def check_availability(self, reservation) -> Availability: ...
    async def prepare(self, reservation) -> CommitPlan: ...    # builds the preview, no side effects
    async def commit(self, plan, *, payment=None) -> BookingResult: ...   # the irreversible step
```
`prepare()` produces the object the confirmation gate shows the user. `commit()` is only ever
called *after* approval.

**`PrivacyCardService`** (`workflows/reservations/payment.py`) — wraps the Privacy.com API
(<https://developers.privacy.com/docs/getting-started>). Creates a **single-use** card
(`type: "SINGLE_USE"`) with `spend_limit_duration: "TRANSACTION"` and `spend_limit` **hard-capped
at $10 (`1000` cents)** — the code refuses to mint any card above $10, full stop. Returns
PAN/exp/CVV to the channel for one transaction; **card data is never logged, never sent to the
LLM, and never persisted to disk** beyond Privacy's opaque card token.
**No escalation:** if a channel reports the establishment needs an authorization/charge above
$10, the workflow **aborts the booking** and surfaces it to the user ("Lazy Bear wants a $25
hold — above my $10 limit; you'll need to handle this one, sir."). It does not mint a larger
card even with confirmation.

**`CalendarService`** (`workflows/reservations/calendar.py`) — on transition to `CONFIRMED`,
creates a Google Calendar event with all booking details (business, address, date/time, party
size, confirmation number, special requests) using Friday's existing Google Calendar tools.
Failure to create the event never fails the booking — it's logged and surfaced.

**`TelegramNotifier`** (`workflows/reservations/notify.py`) — delivers async outcomes
(confirmation, "slot opened — confirm?", call result, failure) over **Telegram** when Friday
isn't actively listening; when it is, the `BackgroundTaskRunner` speaks instead. Backed by
the Telegram Bot API; target number from config.

**Durable persistence + async driver** — provided by the multi-turn framework's
`SqliteSessionStore` and `BackgroundTaskRunner`; the reservation workflow implements `on_tick`
and stores its data in `session.slots`/`session.scratch`. No reservation-specific task store. A
small read model over the session DB can list "your pending reservations" if needed.

---

## 5. Channels in detail

### 5.1 OpenTable / Resy / Yelp (browser automation)
- **Why browser, not API:** OpenTable's API is partner/B2B-only; Resy has none; Yelp
  Reservations booking is likewise not openly available. Driving the real site with the
  user's logged-in account is the only individual-viable path. (Yelp's **Fusion API** is
  still used for *discovery* — see `BusinessDiscovery` — just not for the booking write.)
- Playwright (Chromium), **persistent browser context** per platform stored under
  `~/.friday/browser/<platform>` so the user logs in once (and solves any 2FA) and the
  session is reused. We do **not** store raw passwords.
- `check_availability` reads the slot grid; `prepare` fills the form up to the final submit;
  `commit` clicks confirm. If the site demands a card, control hands to `PrivacyCardService`.
- One thin Playwright base class; OpenTable/Resy/Yelp differ only in selectors/login flow.
- **Risk controls:** human-like pacing, no CAPTCHA-solving services, respect rate limits,
  honor a kill-switch env var. Acknowledge ToS risk (§8).

### 5.2 GenericWeb (our Playwright engine) — *preferred no-API path*
- A single maintained engine that, given a booking URL, uses the LLM to map page fields →
  reservation slots, fills them, and stops at the submit/confirm step for `prepare`.
- This is the **primary** no-API mechanism (your choice: "prefer ours").

### 5.3 SandboxBot (third-party GitHub bot) — *fallback only*
- Triggered only when GenericWeb fails on a site.
- Flow: search GitHub (e.g. `<restaurant platform> reservation bot`) → vet repo (stars,
  recent activity, lockfile present, an allowlist of permitted languages/runtimes) →
  **clone into an isolated Docker container** with **no host filesystem mounts, no host
  network beyond the target domain, no access to host env/secrets** → run → capture result →
  **tear down the container and delete the clone**.
- The container is the trust boundary. Credentials (including the Privacy card) are passed in
  narrowly and only when the booking step needs them, never the full environment.
- Requires Docker on the host; if Docker is unavailable, this channel is disabled and the
  agent falls back to phone.

### 5.4 Phone (AI voice — Bland.ai)
- **Backend: Bland.ai.** We send Bland a call request with the destination number and a
  task/pathway prompt; Bland places the call, runs the conversation, and returns a transcript +
  structured outcome (via webhook or polling). No telephony/media plumbing on our side.
- The LLM generates a **call plan** (objective, date/time, party size, name, callback number,
  special requests, acceptable fallback times) which becomes Bland's task prompt. Per the
  autonomy choice, **the user approves the call plan before we dial.**
- The call outcome lands asynchronously, so the session goes to `WAITING` and is advanced by
  `on_tick` when Bland's result arrives. The call plan/pathway instructs the agent to **handle
  the common branches in-call** and report a **structured outcome**:
  - **`confirmed`** (+ negotiated time, any confirmation #) → `CONFIRMED`.
  - **`no_availability`** — they have no spot, even across the user's flexible times (the call
    agent is told to ask about alternative times/dates within the stated flexibility *before*
    giving up). → see "no-availability handling" below.
  - **`email_requested`** — they ask to be emailed. The agent **captures the email address and
    any instructions** during the call. → hand off to the **Email channel** (§5.5): draft the
    request, get user approval, send, watch for the reply.
  - **`callback_required`** — they'll call back, or said to call a specific later time. →
    `WAITING` with `wake_at` set accordingly (or note the business will call the user's number).
  - **`needs_info`** — they asked something not in the call plan. → surface the question to the
    user (speak/Telegram), collect the answer, re-call.
  The outcome is announced (spoken or Telegram).
- **No-availability handling.** Treated like `UNAVAILABLE`: the agent reports back and offers
  options — try another date/time, call back another day (wait-and-retry via `WAITING`/`on_tick`,
  bounded by `RESERVATION_CALL_DEADLINE_DAYS`), try a different business, or stop. If Friday
  isn't listening, the options go out over Telegram and the session waits for the user's choice.
  Nothing is re-attempted without the user picking an option.
- **Off-hours deferral.** Before dialing, the channel checks the business hours/timezone
  captured during discovery against `now`:
  - **Open now** → place the call immediately.
  - **Closed now** → the session goes to `WAITING` with `wake_at` = the **next opening time**
    (next interval in `hours`, accounting for the business's timezone, special/holiday hours,
    and the requested reservation date — never schedule a call for *after* the reservation it's
    trying to make). The user is told up front: *"They're closed now, sir — I'll ring them when
    they open at 9 a.m. tomorrow."* The call plan is approved **once, now**, so no re-confirmation
    is needed at open time.
  - At `wake_at`, `on_tick` places the approved call. If hours data is missing/ambiguous, the
    agent asks the user for a time window instead of guessing.
- **Retry policy.** No answer / busy / closed-early → retry a few times within the same open
  window (configurable), then defer to the next window; after a deadline, give up and notify.
- **Google Voice is not supported** (no call API). Optional fallback mode `dial_and_bridge`:
  place the call and connect the user in to speak themselves.
- Compliance note: state that it's an automated assistant; call recording / two-party-consent
  laws vary by state (§8).

### 5.5 Email (reservation-by-email)
- **Entered two ways:** (a) a phone call returns `email_requested` (address captured in-call),
  or (b) discovery classifies the business as email-only (`method: email`, address from the
  site/Yelp).
- **Compose:** the LLM drafts a polite reservation-request email from `session.slots` (business,
  service, date/time + flexibility, party size, guest name, callback phone, special requests).
- **Approve before send:** sending is an outward-facing commit → the draft (recipient, subject,
  body) goes through `AWAITING_CONFIRMATION`; the user can edit or approve. **Nothing is sent
  without approval.** Sent from the user's own mailbox via Gmail API or SMTP (config §7).
- **Await reply:** after sending, the session goes to `WAITING`; `on_tick` polls for a reply
  from that business (matched by sender/thread) up to `RESERVATION_EMAIL_DEADLINE_DAYS`. An LLM
  classifies the reply as **confirmed / declined / needs-info / asks-to-call**:
  - confirmed → `CONFIRMED` (calendar event + notify);
  - declined → no-availability handling (offer alternatives);
  - needs-info → surface to user, reply after approval;
  - asks-to-call → route to the Phone channel.
- A one-time **nudge/follow-up** may be sent after a configurable wait if there's no reply, then
  the agent gives up and notifies.
- Only the reservation thread is read — the agent does **not** scan the rest of the inbox (§8).

---

## 6. Reservation state machine

This FSM **is** the `session.fsm_state` from the multi-turn framework; each transition is driven
by a `TurnResult` the workflow returns. `AWAITING_CONFIRMATION` maps to `TurnControl.AWAIT_CONFIRMATION`,
`WAITING` to `TurnControl.BACKGROUND` (advanced by `on_tick`), and `CONFIRMED`/`CANCELLED` to
`COMPLETE`/`CANCEL`.

```
NEW
 └▶ DISCOVERING            (BusinessDiscovery running)
     └▶ CHANNEL_SELECTED   (ChannelDecision made)
         └▶ COLLECTING     (slot-filling; ask user only for what the channel needs)
             └▶ CHECKING_AVAILABILITY
                 ├▶ AVAILABLE ─────────────▶ AWAITING_CONFIRMATION
                 └▶ UNAVAILABLE
                     ├▶ (user opts to wait) ▶ WAITING ──(slot opens)──▶ AWAITING_CONFIRMATION
                     └▶ (no) ──────────────▶ CANCELLED
 AWAITING_CONFIRMATION
   ├▶ (needs card ≤ $10) ▶ MINTING_CARD ─▶ AWAITING_CONFIRMATION (show card will be used)
   ├▶ (needs card > $10) ▶ ABORTED (hand back to user — hard $10 cap, never minted)
   ├▶ user: yes ──┬▶ (phone + closed now) ▶ WAITING ──(business opens / on_tick)──┐
   │              │                                                              │
   │              └▶ (open now / online) ───────────────────────────────────────┤
   │                                                                             ▼
   │                                                                        COMMITTING
   └▶ user: no  ───▶ CANCELLED                                                   │
                                                                                 ▼
   COMMITTING outcomes:
     ├▶ confirmed ─────────▶ CONFIRMED
     ├▶ no_availability ───▶ UNAVAILABLE (offer: other time / retry later / other business / stop)
     ├▶ email_requested ──▶ AWAITING_CONFIRMATION (approve draft) ▶ EMAIL_SENT ▶ WAITING (reply)
     │                         reply: confirmed→CONFIRMED · declined→UNAVAILABLE ·
     │                                needs_info→ask user · asks_to_call→Phone
     ├▶ callback_required ─▶ WAITING (wake_at = stated time / business will call)
     ├▶ needs_info ────────▶ ask user (speak/Telegram) ─▶ re-COMMIT
     └▶ failed (no answer)─▶ retry within window / defer to next open window / give up
 CONFIRMED ─▶ store confirmation #  ─▶ create Google Calendar event  ─▶ notify (speak / Telegram)
```

Email-only businesses enter the `EMAIL_SENT ▶ WAITING (reply)` path directly from
`CHANNEL_SELECTED` rather than via a call.

Slots: `business_name`, `service_type` (e.g. dinner, haircut, 60-min massage), `date`,
`time` (+ flexibility window), `party_size` / `guests`, `guest_name`, `phone`, `email`,
`special_requests`, `seating_or_staff_pref`, plus discovery-/call-derived `business_phone`,
`business_email`, `business_hours`, `business_timezone`, and `email_thread_id` (used for
off-hours call deferral and email-reply tracking). **Up front** the agent asks only for the
essentials (business, service, date, time, party size); channel-specific extras (email, card)
are requested lazily in `COLLECTING`. `service_type` covers appointment businesses (which
service / which stylist) as well as restaurants.

---

## 7. Configuration (env vars)

Registered in `main.py` only when the feature's key is present, mirroring existing
integrations.

| Variable | Purpose |
|---|---|
| `RESERVATIONS_ENABLED` | Master on/off for the workflow. |
| `PRIVACY_API_KEY` | Privacy.com API key (card minting). |
| `RESERVATION_CARD_LIMIT_USD` | **Hard cap, default `10`.** Code never mints above this; bookings needing more are aborted. (A ceiling, not a default to raise per-booking.) |
| `RESERVATION_PHONE_PROVIDER` | `bland` (default) \| `dial_and_bridge` \| `off`. |
| `BLAND_API_KEY` | Bland.ai API key (autonomous calling). |
| `RESERVATION_CALL_RETRIES` | Retries per open window on no-answer/busy before deferring (default `3`). |
| `RESERVATION_CALL_DEADLINE_DAYS` | Stop trying after this many days of deferral (default `7`). |
| `RESERVATION_EMAIL_PROVIDER` | `gmail` (Gmail API, user's mailbox) \| `smtp` \| `off`. |
| `RESERVATION_EMAIL_FROM` / SMTP host/port/user/pass (or Gmail OAuth) | Sending identity/credentials. |
| `RESERVATION_EMAIL_DEADLINE_DAYS` | Stop awaiting an email reply after this many days (default `5`); one follow-up nudge before giving up. |
| `YELP_API_KEY` | Yelp Fusion API key (business discovery). |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_ALLOWED_CHAT_IDS` / `TELEGRAM_NOTIFY_CHAT_ID` | Telegram notifications when not listening. |
| `RESERVATION_CALENDAR_ID` | Google Calendar to write confirmed events to (default `primary`). |
| `RESERVATION_USER_PHONE` / `RESERVATION_USER_EMAIL` / `RESERVATION_GUEST_NAME` | Defaults for slot-filling. |
| `RESERVATION_ALLOW_SANDBOX_BOTS` | Default `false`; enables the Docker GitHub-bot fallback (Docker is available). |
| `RESERVATION_BROWSER_DIR` | Persistent Playwright profile dir (default `~/.friday/browser`). |
| `RESERVATION_KILL_SWITCH` | If set, all commits/calls are blocked (research-only). |
| `TAVILY_API_KEY` | Already used by search; reused for discovery fallback. |

Durable booking/session state lives in the multi-turn framework's session DB
(`~/.friday/sessions.db`); there is no separate reservation task store.

---

## 8. Security, safety & compliance

### 8.1 Controls

- **Human-in-the-loop on every commit:** no booking, call, or charge without an explicit,
  summarized confirmation.
- **Payment containment:** single-use card, **hard $10 cap** enforced in code (mint refused
  above $10; over-$10 bookings aborted). Card PAN/CVV is **never logged, never sent to the LLM,
  never written to disk**; passed to a channel only at the moment of charge, and scoped
  narrowly into the sandbox if a bot is used.
- **Untrusted code containment:** GitHub bots run only in Docker with no host mounts, no
  secrets in env, **egress restricted to the target domain via an allowlist proxy**; clone +
  container destroyed after. Disabled by default (`RESERVATION_ALLOW_SANDBOX_BOTS`).
- **Credentials:** browser sessions stored as a persistent Playwright profile (cookies/tokens),
  not plaintext passwords; API keys via env only.
- **Kill switch:** `RESERVATION_KILL_SWITCH` forces research-only mode.

### 8.2 Private-data leak surfaces (where PII can escape *without the user noticing*)

The HITL gate covers *actions* (book/call/charge), but data flows during the **research phase**
before any confirmation — that's where silent leakage hides. Each surface below has a required
mitigation. **Important honesty:** the $10 single-use card bounds the damage from a *leaked card
number* only. It does **not** mitigate a leaked `PRIVACY_API_KEY` (an attacker could mint many
cards against your funding source) or leakage of *personal* data (name, phone, email, account
sessions). Those need their own controls.

| # | Surface | What could leak | Required mitigation |
|---|---------|-----------------|---------------------|
| L1 | **Conversation/debug logs** — existing `log_conversations`/`conversations.log` (`config/settings.py:222`) and `_log("Processing: {text}")` (`core/assistant.py:125,237`) print/store raw utterances | name, phone, email, special requests; potentially card text | Redact PII + card patterns before any log/print; **never** log card data; disable conversation logging for reservation sessions by default; log file `0600`. |
| L2 | **Session DB** `~/.friday/sessions.db` (slots: name/phone/email) | personal contact details at rest | Store only Privacy's opaque card token (never PAN/CVV); file/dir perms `0600`/`0700`; purge PII on completion after a short retention window; consider at-rest encryption. |
| L3 | **Browser profile** `~/.friday/browser` (OpenTable/Resy/Yelp session cookies) | account-takeover tokens | `0700` perms; gitignored; documented as sensitive; never copied into the sandbox. |
| L4 | **LLM provider** (Anthropic/OpenAI) — slot extraction, call-plan generation | name/phone/email/reservation details sent to a 3rd party | Disclose in setup that Claude is used; **never include card data** in any prompt; offer local Ollama for the most sensitive extraction; don't paste raw account cookies/secrets into prompts. |
| L5 | **Discovery queries** (Yelp Fusion, Tavily) | a naive query could include user PII | Hard rule: discovery queries contain **only** business name + location, never the user's name/phone/email/card. |
| L6 | **Bland.ai** — receives destination number + call plan (name, callback #, details); processes/records call audio | user's name + callback number to a 3rd party; possibly card read aloud if a phone booking needs payment | Minimize: first name + callback number + reservation specifics only. **If a phone booking would require reading a card aloud, abort** (don't voice card data over the line). Review Bland's retention; disclose in setup. |
| L7 | **Google Calendar event** | full booking details, confirmation code | Write to the user's own (private) calendar (`primary` default); default visibility private; don't embed card data. |
| L8 | **Telegram notifications** | booking details | Send only to the user's own configured `TELEGRAM_NOTIFY_CHAT_ID` chat. |
| L9 | **`PRIVACY_API_KEY` / other API keys** | mint many cards / impersonate services | Env only, never logged, never in prompts, gitignored (`.env` is); recommend account-level spend limits at Privacy so a key leak is also bounded, not just per-card. |
| L10 | **Sandbox exfiltration** (L4-class but worse) | card + PII handed to untrusted code | Pass the minimum PII; rely on egress allowlist + single-use $10 card; **require explicit user consent before using a third-party bot** for a given booking (new trust surface, off by default). |
| L11 | **Email channel** — sends from the user's mailbox; reads replies | reservation email (name/contact/details) to the business; inbox access | Send only after user approves the draft; **read only the reservation thread** (match by recipient/thread-id), never scan the broader inbox; OAuth/SMTP creds via env, never logged; never include card data in the email. |

**Default posture:** PII redaction in logs is on by default; the data shared with each third
party (LLM, Bland, Yelp, Tavily, Google) is disclosed in setup docs so nothing leaves the
machine that the user hasn't been told about.

### 8.3 Legal / ToS realities to accept before build
- Automating OpenTable/Resy/Yelp may violate their Terms of Service and risks account suspension.
- Automated phone calls and call recording are regulated (TCPA; two-party-consent states). The
  agent should state it's an automated assistant and avoid recording where prohibited.
- Running third-party code is inherently risky even sandboxed.
These are product/legal acceptances, not things code can fully neutralize.

---

## 9. Proposed file layout

```
workflows/reservations/
├── __init__.py
├── workflow.py            # ReservationWorkflow — a ConversationalWorkflow (start/resume/on_tick)
├── models.py              # Reservation, Slot, ChannelDecision, CommitPlan, BookingResult
├── discovery.py           # BusinessDiscovery (Yelp Fusion + Tavily + LLM classifier)
├── router.py              # ChannelRouter (strategy selection + preference order)
├── payment.py             # PrivacyCardService
├── calendar.py            # CalendarService (Google Calendar event on CONFIRMED)
├── notify.py              # TelegramNotifier
└── channels/
    ├── base.py            # ReservationChannel ABC + shared Playwright base
    ├── opentable.py       # Playwright + persistent session
    ├── resy.py            # Playwright + persistent session
    ├── yelp.py            # Playwright + persistent session
    ├── generic_web.py     # our LLM-driven Playwright form engine (preferred no-API path)
    ├── phone.py           # Bland.ai; structured-outcome handling; dial_and_bridge fallback
    ├── email.py           # EmailChannel: draft → approve → send (Gmail/SMTP) → poll for reply
    └── sandbox_bot.py     # Docker-isolated GitHub bot fallback
```
Session state, persistence, and the background runner come from `core/conversation/` (multi-turn
framework) — **not** duplicated here. Plus: `ReservationConfig` dataclass in
`config/settings.py`, conditional registration in `main.py:create_workflow_manager`, and new
deps in `requirements.txt` (`playwright`, `docker`; HTTP for Bland.ai/Yelp/Telegram). The
`core/assistant.py` change and `turnstile-ctx` dependency are owned by the multi-turn spec.

---

## 10. Phased delivery

> **Prerequisite:** the multi-turn framework's **MT1–MT2** (context register + session core +
> `ConversationalWorkflow`) must land first; **MT3** (background runner) is required for M5
> (wait-and-book) and async phone results. See `multi-turn-agent-spec.md` §10.

1. **M1 — Skeleton + discovery.** ✅ **Done.** `ReservationWorkflow` (a `ConversationalWorkflow`),
   `BusinessDiscovery` (Yelp Fusion + Tavily + heuristic classifier), and `models.py`
   (`ChannelDecision`, `ReservationMethod`) under `workflows/reservations/`; registered in
   `main.py` behind `RESERVATIONS_ENABLED`. Collects essentials over multiple turns, looks the
   business up, classifies the method (OpenTable/Resy/Yelp/generic-web/phone/email/unknown), and
   **describes the plan — no commits**. Tested in `tests/test_reservations.py` (discovery
   classification incl. phone fallback + card hint; slot extraction; full + drip-fed flows).
   **LLM refinement landed (live-env work):** `llm.py` (`ReservationLLM`, one-shot JSON
   completions on `RESERVATION_LLM_MODEL`, default Haiku) now backs slot extraction
   (handles "four of us", "half past seven"; regex baseline kept as fallback) and discovery
   classification (incl. email-only detection, anti-hallucination checks — URLs/emails must
   appear in the evidence; the OpenTable/Resy domain match is never overridden). Heuristics
   remain the offline fallback; verified live (Tavily + Haiku correctly classified a Tock
   prepaid-ticketing restaurant as `generic_web` + card hint).
2. **M2 — Channels + confirmation gate.** ✅ **Done (framework + gate; live selectors are
   best-effort).** `ReservationChannel` ABC + `Availability`/`CommitPlan`/`BookingResult`
   (`channels/base.py`), a shared `BrowserChannel` (Playwright lifecycle, optional dep) with
   `GenericWebChannel` (real best-effort form-filler) and `OpenTable`/`Resy`/`Yelp` subclasses,
   plus `ChannelRouter`. The workflow now selects a channel, checks availability, and **gates on
   explicit user approval** before `commit()`. Honored: kill switch (research-only), card-required
   → safe hand-off (payment is M3), Playwright-missing → safe hand-off. Tested in
   `tests/test_reservations.py` (confirm→book, decline, unavailable, kill switch, phone-describe,
   drip-fed→gated, channel commit gates). **Note:** per-site logged-in booking flows for
   OpenTable/Resy/Yelp need real accounts + live selector tuning (and carry ToS risk); until then
   they hand off rather than fake success — the orchestration, gate, and GenericWeb engine are in
   place.
3. **M3 — Privacy.com payment.** ✅ **Done.** `PrivacyCardService` (`payment.py`) mints
   **single-use** cards, **hard-capped at $10** in code (over-cap mints refused; configured limit
   can only lower the ceiling). Wired into the confirmation→commit path: when a booking needs a
   card, the workflow mints just before `commit()` and passes it as `payment`; mint failure aborts
   the booking. `VirtualCard.__repr__` redacts PAN/CVV; card data is never logged, persisted, or
   put in slots. Tested (hard-cap refusal, mint→passed-to-commit, mint-failure abort, no-service
   hand-off). Entering the card into the booking page is part of per-site live work (M2 caveat).
4. **M4 — Calendar + Telegram.** ✅ **Done.** `CalendarService` (`calendar.py`) builds a Google
   Calendar event (summary/location/description + parsed date/time) on a confirmed booking;
   insertion is pluggable (real Google inserter only when `GOOGLE_APPLICATION_CREDENTIALS` is
   set, else a safe no-op). `TelegramNotifier` (`notify.py`) sends a written confirmation via
   the Telegram Bot API. Both are wired into the confirmed-commit path and **never fail the
   booking** on error. Tested: event-dict build + date/time parse, no-op-without-inserter,
   unparseable-time skip, Telegram payload + never-raises, and end-to-end (confirm → calendar +
   Telegram fired). Real Google OAuth/credentials setup remains environment config (Friday has no
   Google auth wired in yet).
5. **M5 — Phone channel (Bland.ai).** ✅ **Done.** `PhoneChannel` + `BlandClient`
   (`channels/phone.py`): builds a call plan, places the call after the user approves it (the
   gate), then the session goes `WAITING` and the workflow's `on_tick` (MT3 runner) polls Bland
   and maps a **structured outcome** (confirmed / no-availability / email-requested / callback /
   needs-info / failed). Off-hours calls **defer to the next opening** (`is_open_now` over Yelp
   hours); no-answer **retries** up to `RESERVATION_CALL_RETRIES` then gives up. Confirmed →
   calendar + Telegram (M4). `DialAndBridgeChannel` is the non-autonomous fallback; Google Voice
   unsupported. Router wires `PHONE` from `RESERVATION_PHONE_PROVIDER` (`bland`/`dial_and_bridge`/
   `off`). Bland client injectable; tested (outcome classification, hours/deferral, router wiring,
   full confirm→call→poll→confirmed flow, no-availability). Live calling needs a `BLAND_API_KEY`.
6. **M5b — Email channel.** ✅ **Done.** `EmailChannel` (`channels/email.py`): template drafter
   (LLM-injectable) → **editable approval gate** (`confirm_email`: "yes" sends, "no" cancels,
   anything else re-drafts) → send (SMTP via stdlib; reader/sender injectable) → session `WAITING`
   → `on_tick` polls **only the reservation thread** (subject-scoped) → classifies the reply
   (confirmed / declined / needs-info / asks-to-call), with a deadline. Confirmed → calendar +
   Telegram (M4). Wired both ways: a phone `email_requested` outcome **hands off** to an email draft
   (promotes the WAITING session back to an active confirmation), and email-only businesses route
   here. Router wires `EMAIL` from `RESERVATION_EMAIL_PROVIDER`. Tested: reply classification,
   drafter content, send+poll, router wiring, full phone→email→send→reply→confirmed, and the
   draft-edit path. **Email-only discovery now routes here directly** (both the heuristic
   "email us to reserve" detector and the LLM classifier emit `method: email`), and the
   drafter is LLM-backed when available (template fallback kept). Real SMTP/IMAP still need
   env config (see `.env.example`).
6. **M6 — Wait-and-book.** ✅ **Done.** When a channel reports a slot `UNAVAILABLE`, the agent
   offers to watch (`confirm_wait`); on "yes" the session goes `WAITING` and `_tick_watch`
   re-checks availability each interval (`RESERVATION_WATCH_INTERVAL_SECONDS`). When a slot
   opens it **re-gates for approval** ("a table just opened — shall I book it?") and routes
   into the normal commit path (so the HITL rule still holds); after
   `RESERVATION_WATCH_DEADLINE_DAYS` with nothing, it gives up and notifies. Tested: offer +
   decline, watch→opens→re-gate→book, and deadline give-up. (Effective once a channel's
   `check_availability` returns concrete AVAILABLE/UNAVAILABLE — browser channels currently
   return UNKNOWN, the per-site availability work noted in M2.)
7. **M7 — Sandboxed GitHub-bot fallback.** ✅ **Done (scaffolding + gating; live execution
   best-effort).** `SandboxBotChannel` + `GitHubBotFinder` (vets by stars / recent activity /
   allowed language / not-archived, via `gh`) + `DockerSandbox` (`channels/sandbox_bot.py`).
   Triggered only when our own automation returns `needs_manual`, **off unless
   `RESERVATION_ALLOW_SANDBOX_BOTS` is set and Docker is present**, and gated behind a **separate,
   repo-named consent** ("I found acme/ot-bot (120★) — third-party code, run it sandboxed?").
   The container clones the repo itself (no host mounts), runs with `--network none`, dropped
   caps, read-only FS, no-new-privileges, pids/memory/cpu limits, no host env/secrets, a timeout,
   and `--rm` teardown; only minimal booking facts are passed in (never credentials). Finder +
   sandbox runner injectable; tested (vetting, default-off, fallback→consent→run, decline,
   no-sandbox hand-off). **Honest limit:** generically driving an arbitrary repo to *complete* a
   booking isn't reliably solvable or fully safe — the harness runs best-effort and hands off when
   it can't confirm; live container execution is unverified here.

8. **M8 — TableCheck standing watch.** ✅ **Done.** A notify-only, multi-criteria availability
   watcher for venues on TableCheck (driving case: Bar BenFiddich, `benfiddich-tokyo`), built
   on the existing session/runner machinery — no new process, no new store.
   - **Channel** (`channels/tablecheck.py`): pure-HTTP `TableCheckChannel` (aiohttp, no
     browser/login) — the first channel whose `check_availability` returns concrete
     AVAILABLE/UNAVAILABLE, which also makes the M6 wait-and-book path live for TableCheck
     bookings. `fetch_month(slug, party_size, month)` is the watcher's polling primitive
     (TableCheck availability is party-size-dependent, so each distinct party size is its own
     fetch unit). `commit()` is a safe hand-off with a reserve-page deep link (no auto-booking).
     The endpoint template is env-overridable (`TABLECHECK_AVAILABILITY_URL`) and responses are
     strictly schema-validated — an unrecognised shape raises `SchemaDrift`, which **pauses the
     watch and alerts** instead of silently reporting "no availability" (endpoint drift is the
     #1 failure mode; fixtures in `tests/fixtures/tablecheck/` make re-adaptation fast).
     ⚠️ The default endpoint/params encode the documented web_booking pattern and still need
     one manual verification against live widget devtools traffic before production reliance.
   - **Pure core** (`watcher.py`): `WatchCriterion` (date range × acceptable times × party
     size, independent lifecycle: active/fulfilled/expired), fetch-unit planning, the snapshot
     differ (`slot_opened/closed`, `calendar_published`, `venue_halted/resumed`; first snapshot
     is a baseline), the event×criteria matcher, and conversational date-range/time parsing.
     Side-effect-free, like `snipe.py`.
   - **Workflow**: "Watch BenFiddich for September 5th or 6th at 7 or 9pm for 2" →
     gather → `confirm_watchlist` gate → `watching_list` (WAITING). `_tick_watchlist` polls
     each fetch unit (≥`TABLECHECK_UNIT_SPACING_SECONDS` apart), diffs against the last
     snapshot (kept in `session.scratch`), and Telegram-notifies matches with a booking deep
     link (30-min dedupe per slot; close→reopen re-notifies; one summary on calendar-publish
     day instead of per-slot spam). Cadence: normal 300s ± jitter, hourly idle while the
     calendar is closed, 60s burst around a known drop; hard floor 60s; exponential backoff on
     429/5xx with an alert after 3 consecutive failures. Criteria auto-expire venue-local.
   - **Release-time research**: when the calendar isn't published and no drop time is known,
     the watch reuses `resolve_release_policy` (web + Reddit + LLM, M-snipe Phase 2) once per
     watch, converts a credible policy (≥ `RESERVATION_SNIPE_MIN_CONFIDENCE`) into a burst
     window via `compute_release_fire_ts`, self-schedules it, and tells the user what it found;
     a user-stated policy ("they open 25 days out at 9am JST") does the same at confirmation.
   - **Control verbs**: WAITING sessions never take a user turn, so "stop watching X" /
     "any luck with the watch?" / "we got the table" arrive as new turns and act on standing
     watches **through the shared session store** (injected by the assistant); a second
     "watch <same venue>" ask folds new criteria into the existing watch session.
   Tested end-to-end with a fixture-driven fake fetcher: gather/confirm/decline, non-TableCheck
   refusal, baseline + notify + dedupe + reopen, unmatched-slot silence, multi-party fetch
   units, backoff + drift-pause, halted/resumed, publish summary, research → burst scheduling,
   low-confidence idling, manual policy, publish-beats-prediction, and all control verbs
   (`tests/test_watcher.py`, `tests/test_tablecheck.py`).

Each milestone is independently shippable and gated behind env vars, so partial rollout never
destabilizes existing Friday behaviour.

---

## 11. Resolved decisions

All v1 scoping questions are settled:

1. **Scope** — appointment-based businesses (restaurants, hair salons, massage/spa, similar
   bookable services). Hotels out of scope.
2. **Phone backend** — **Bland.ai** (autonomous, user-approved call plan); `dial_and_bridge`
   fallback.
3. **Accounts** — user has a Privacy.com API key and OpenTable / Resy / Yelp logins; **Yelp
   integrated** (Fusion API for discovery, browser for booking).
4. **Docker** — available locally; the sandboxed GitHub-bot fallback (M7) is enabled.
5. **Calendar** — confirmed bookings create a Google Calendar event with all details.
6. **Async notifications** — **Telegram** message when Friday isn't listening; spoken when it is.
7. **Multi-user / concurrency** — single user, one active dialogue (per multi-turn spec).
