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
.venv/bin/python -m research status          # counts, artifacts, last run
.venv/bin/python -m research nightly --dry-run   # loop without training/API calls
.venv/bin/python -m research nightly         # real run (trains, pays for judging)
.venv/bin/python -m research rate            # rate production-vs-shadow pairs
.venv/bin/python -m research revert --arm lora --to v20260718

# 4. When trusted, schedule it (03:30 daily; keep the Mac awake for it):
research/scripts/install_nightly.sh
```

## Costs and storage

- Paid API per real nightly run: Sonnet judging (2 calls per replayed exchange
  per arm), one prompt-evolution call, one call per new mined correction.
  Everything else is local (mlx_lm training/generation, Ollama, ChromaDB).
- `~/.friday/research.db` — append-only conversation/feedback/eval record;
  `~/.friday/research/<arm>/vYYYYMMDD/` — versioned artifacts, `current`
  pointer file selects the active one. Only aggregates are committed
  (`results/eval.csv`, `results/nightly/*.md`).

## Env vars

- `FRIDAY_RESEARCH=1` — master opt-in for recording/buttons/shadow.
- `ANTHROPIC_API_KEY` — judging, prompt evolution, correction synthesis.
