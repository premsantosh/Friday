# LangGraph agent engine

Opt-in orchestration core for Friday: `FRIDAY_AGENT_ENGINE=langgraph`. Default stays the legacy router, and any engine error falls through to it, so Friday works identically with the flag off.

## What it changes

| Before | After (flag on) |
|---|---|
| Chat history: in-RAM list, 10 turns, lost on restart | LangGraph checkpoint per user thread in `~/.friday/agent_checkpoints.db`, trimmed to `AgentConfig.history_max_messages`, survives restarts |
| Step 3: one-shot JSON intent classifier (`llm/router.py`) | Native tool calling over the workflow registry; multi-step chains in one request |
| Confirmations: only inside legacy sessions | `interrupt()`-based confirmations that survive restarts and channel switches; answered by the next turn (deterministic yes/no parsing) |
| No self-scheduling | `schedule_wakeup` tool + `agent_wakes` table serviced by `BackgroundTaskRunner` |

Unchanged: active legacy sessions (Step 0), keyword match (Step 1), intent cache (Step 2), the reservations workflow (stays a legacy `ConversationalWorkflow`, reachable via the `start_task` handoff tool), the harness (`ActionGate`, egress sinks, audit log).

## Turn priority (`VoiceAssistant.process_input`)

1. Active legacy session owns the turn (global escape cancels it).
2. Pending agent confirmation owns the turn (global escape abandons it). This sits before keyword matching on purpose: "yes please" must answer the question, not match a workflow.
3. Keyword match → 4. intent cache → 5. agent engine → on error, legacy router.

## Module map (`agent/`)

| File | Role |
|---|---|
| `__init__.py` | guarded imports; `is_available()` |
| `state.py` | `AgentState` TypedDict (messages, user_id, context_block, handoff_message, tool_iterations, fact-key feedback) |
| `models.py` | `LLMConfig` → `ChatAnthropic` / `ChatOpenAI` / `ChatOllama` |
| `tools.py` | tools from the `WorkflowManager`: simple workflows, `start_task` handoff, gated wrappers (`GateSpec`, `GATE_SPECS`), `schedule_wakeup` |
| `nodes.py` | prepare_context → agent → tools / overflow → finalize; pair-aware trim |
| `graph.py` | uncompiled `StateGraph`; compiled per invocation |
| `checkpoint.py` | `AsyncSqliteSaver` opened per invocation (loop-local), `InMemorySaver` in ephemeral mode |
| `store.py` | `agent_threads` (epochs for `reset()`) and `agent_wakes` tables, plain sqlite3 |
| `tracing.py` | optional LangSmith tracer with PII redaction + hidden context block |
| `engine.py` | `AgentEngine` facade: `handle`, `has_pending_interrupt`, `cancel`, `reset`, `run_due_wakes` |

## Gated actions

`GateSpec.requires_confirmation(intent, entities)` decides per call; the tool builds the plan dict, calls `interrupt()` with the question, and after a "yes" executes through `ActionGate` (kill switch `FRIDAY_KILL_SWITCH`, approval hash of the exact plan shown, idempotency via the audit log keyed by the tool call id). Everything before `interrupt()` is pure because LangGraph re-runs the tool function on resume. Proving ground: `hass_locks` unlock (`ActionKind.DEVICE_CONTROL`).

Caveats to know about:
- Keyword matching (Step 1) runs before the engine, so a plainly phrased "unlock the back door" still executes `hass_locks` directly as it always has. The gate protects agent-initiated calls (multi-step chains, paraphrases the keywords miss). Moving the gate in front of Step 1 is a separate decision.
- Only `hass_locks` has a `GateSpec` today; every other workflow is bound ungated, exactly as the legacy router executes it. Add entries to `GATE_SPECS` to gate more.
- Gated successes are never written to the intent cache (a cache hit would bypass the confirmation).
- Tool calls run one at a time (`parallel_tool_calls=False`) so an interrupt can't replay a side-effecting sibling.

## Failure containment

- A run that raises *after* a tool executed is not re-raised (the legacy fallback would re-execute the request): the engine closes any orphaned tool call, records an in-character apology on the thread and returns it. Only failures before any side effect reach `VoiceAssistant`'s legacy fallback.
- Orphaned `tool_use` blocks (a tool step cancelled by Ctrl+C / channel shutdown) are repaired at the start of the next turn with an `ERROR:` tool result, so the provider never rejects the thread.
- `schedule_wakeup` is only offered when a `BackgroundTaskRunner` can fire it (not in `--chat`/`--test`). A wake whose thread has a pending confirmation is postponed each tick until the confirmation is answered or lapses.
- Step 0b (`has_pending_interrupt`) costs one SQLite read per turn before keyword matching; measured well under 10 ms locally.

## Event loops and SQLite

Telegram, Voice PE and the background runner each own a private asyncio loop; aiosqlite connections are loop-bound. The engine therefore opens `AsyncSqliteSaver.from_conn_string()` as a context manager per invocation over one WAL-mode file. `agent_threads`/`agent_wakes` use a plain sqlite3 connection with a lock (same idiom as `SqliteSessionStore`). Ephemeral mode (`--chat`, `--test`) keeps everything in memory.

## Thread ids

`chat:{user_id}:{epoch}`. Voice/terminal use `default`, Telegram the chat id, Voice PE the room name, so today these are separate conversations. `clear` (or `clear_history()`) bumps the epoch instead of deleting checkpoints.

## Tracing (optional, free tier)

`FRIDAY_LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`. Free tier is 5k traces/month, 14-day retention; with no card on file overage is dropped, never billed. `LANGSMITH_TRACING_SAMPLING_RATE` throttles. Traces leave the machine: `redact_text` masks cards/e-mails/phones and the memory context block is hidden entirely; user text, tool calls and replies remain visible. The tracer is a callback, so swapping to a local backend (Phoenix, Langfuse) touches only `agent/tracing.py`.

## Tests

`tests/test_agent_engine.py`, `test_agent_tools.py`, `test_agent_hitl.py`, `test_agent_background.py`, `test_agent_tracing.py` (fakes in `tests/agent_fakes.py`). No network: a scripted chat model stands in for the LLM; `conftest.py` strips `FRIDAY_AGENT_ENGINE`, `LANGSMITH_*`, `FRIDAY_KILL_SWITCH`.

## Deferred

- Reservations as a subgraph with the harness FSM as validator (design only, not planned).
- Streaming replies (`AgentEngine` returns `str`; `astream` can be added without graph changes).
- Per-loop saver caching if the per-invocation connection ever shows in latency.
- Legacy-path tracing via `langsmith.wrappers.wrap_anthropic` (burns quota faster; separate flag).
