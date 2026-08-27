# introspection — how Friday knows about itself

This package is the single substrate behind the `self_status` workflow
("Did the LoRA run last night?", "run a self-diagnosis"), the
`python -m research doctor` CLI, and the nightly failure alert. Coverage is
**discovered, not enumerated** — a new ability gets reported automatically as
long as it follows the house conventions.

## The extension contract

When you add a new ability to Friday, self-awareness picks it up like this:

| You do | Friday automatically |
|---|---|
| Register a `Workflow` (the only way an ability enters Friday) | lists it in "what can you do?" and the router/agent tool descriptions |
| Keep state under `~/.friday` (a `*.db`, a log in `research/logs/`, an artifact arm dir with `current`/`v*` versions) | inventories it in status, checks it opens read-only, warns on unbounded logs, validates artifact pointers |
| Execute irreversible actions through the harness `ActionGate` | includes them in "what did you do today?" via the audit log |
| Schedule via launchd with a `com.friday.*` label | reports the job and whether it is loaded |
| Implement `status_snapshot()` / `health_checks()` on your workflow (optional, both default to no-op) | merges your own state into "what's your status?" and your own checks into every self-diagnosis |
| Build a subsystem that is not a workflow (a new channel, a scheduler) | needs one small `StatusProvider` registered next to its code — see below |

`tests/test_introspection_extensibility.py` guards this table: registered
workflows must appear in capabilities output, new databases/logs/arms must
appear in snapshots without code changes, and workflow hooks must be merged.

## Rules for providers and hooks

- **Read-only.** Never create files or databases; introspection runs in
  ephemeral modes where persistence must not spring into existence. Open
  SQLite with `file:...?mode=ro` URIs and report `{"available": False}` when
  a store is absent.
- **Text-free.** Counts, ids, stage notes, event names, sizes, versions —
  never user text, reply text, or memory facts (the same invariant the
  research `events` table enforces).
- **Never raise.** The aggregators turn a crashing provider into a FAIL
  check, but degrade to WARN/SKIP yourself where you can.

## Adding a StatusProvider

```python
from introspection import StatusProvider, CheckResult, CheckStatus, register_provider

class MyThingProvider(StatusProvider):
    name = "my_thing"

    def snapshot(self, paths, probes):
        return {"available": True, "queue_depth": 3}

    def checks(self, paths, probes):
        return [CheckResult("my_thing.queue", CheckStatus.PASS, "queue drained")]

register_provider(MyThingProvider())
```

Register at import time from your subsystem's module (core providers live in
`introspection/providers.py` and self-register the same way). `probes` carries
injectable environment access (`launchctl`, `http_get`, `now`) so checks stay
testable offline.
