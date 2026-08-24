# Friday learns: personalization research

Head-to-head study of three ways to make Friday learn from every conversation,
on-device, overnight: reflection memory (arm A), nightly LoRA of a local 8B
(arm B), and system-prompt evolution (arm C). Haiku stays the production
brain; a local model answers the same messages silently in shadow, and each
arm's variant is regenerated nightly from exact context snapshots and judged
pairwise against the vanilla local model. The pre-registered definition of
"improved" lives in `data/evalset/PROTOCOL.md`; pinned versions and measured
smoke results in `scripts/PINNED.md`.

## Running it

```bash
# 1. Opt in (never active in --chat/--test). Data collection starts here.
FRIDAY_RESEARCH=1 python main.py --telegram

# 2. Fill in the FILL-IN probes in research/data/evalset/curated.yaml.

# 3. After a few days of chatting:
.venv/bin/python -m research status          # counts, artifacts, last run, recent events
.venv/bin/python -m research nightly --dry-run   # full loop, fake generation+judge, seconds
.venv/bin/python -m research nightly         # real run (trains, pays for judging)
.venv/bin/python -m research nightly --weekly    # also judge the curated split
.venv/bin/python -m research eval --arms lora,prompt --judge sonnet
.venv/bin/python -m research protocol --arm lora  # where it stands vs the bar
.venv/bin/python -m research rate            # rate production-vs-shadow pairs
.venv/bin/python -m research revert --arm lora --to v20260718

# 4. When trusted, schedule it (03:30 daily; keep the Mac awake for it):
research/scripts/install_nightly.sh
```

`nightly --dry-run` runs harvest through report with deterministic fake
generation and the fake judge, so it exercises the whole pipeline in under a
second without loading model weights or paying for anything. It is the
regression harness: if the loop is broken, that run says so.

## Provenance: what the loop learned from, and when

Every signal the loop uses is recorded in the `events` table, and every artifact
carries a `provenance.json` naming exactly what it consumed. The event taxonomy
is documented in `events.py`.

```bash
.venv/bin/python -m research trace --exchange 412        # one turn's whole life
.venv/bin/python -m research trace --artifact lora/v20260814  # what it was built from
.venv/bin/python -m research trace --run 17              # everything one nightly did
.venv/bin/python -m research trace --since 2026-08-01 --event dataset.included
.venv/bin/python -m research trace --exchange 412 --json # machine-readable
```

`--exchange` shows the turn being recorded, shadowed, mined, fed to each arm,
replayed and judged. `--artifact` shows the manifest (which exchanges, which
feedback, which corrections, the dataset hash) plus the build timeline. Events
carry ids, counts and reasons, never user or reply text: that stays in the
tables it is already in, and is joined at read time.

## Costs and storage

- Paid API per real nightly run: Sonnet judging (2 calls per replayed exchange
  per arm), one prompt-evolution call, one call per new mined correction.
  Everything else is local (mlx_lm training/generation, Ollama, ChromaDB).
- `~/.friday/research.db` — append-only conversation/feedback/eval/event record;
  `~/.friday/research/<arm>/vYYYYMMDD/` — versioned artifacts, `current`
  pointer file selects the active one. Only aggregates are committed
  (`results/eval.csv`, `results/nightly/*.md`).
- The event log adds roughly 40 MB a year and is never pruned; the nightly
  backup rotation in `stage_harvest` covers it.
- `~/.friday/research/logs/live.log` — the in-process research log (shadow
  failures, recorder failures). `nightly.{out,err}.log` alongside it.

## Env vars

- `FRIDAY_RESEARCH=1` — master opt-in for recording/buttons/shadow.
- `ANTHROPIC_API_KEY` — judging, prompt evolution, correction synthesis.
