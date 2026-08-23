# Pre-registered evaluation protocol

Frozen before any training run. Changes to this file after the study starts
must be logged in the Changelog below with a reason — results reported in the
paper use the protocol version in force when they were produced.

## Question

Does a personalization arm (memory / lora / prompt) make the local model's
responses better *for this specific user* than the vanilla local base model?

## Comparisons

- Every arm is compared pairwise against `base` (vanilla
  `mlx-community/Qwen3-8B-4bit`, thinking disabled, pinned persona prompt,
  greedy decoding). All arms generate through the same runtime (mlx_lm) with
  identical decode parameters; the arm's artifact is the only delta.
- Production Haiku is reported as a reference ceiling, never as a competitor.

## Judging

- Primary judge: Claude Sonnet (`claude-sonnet-5`), pairwise with both
  presentation orders; the two verdicts average into one prompt score
  (win 1, tie 0.5, loss 0). Judge errors score as ties.
- Agreement audit: llama3.1 (a different family from the judged Qwen3 base,
  keeping the audit independent) re-judges a 20% sample; direction-agreement
  on decisive prompts is reported alongside every result.
- Human anchor: ~20 pairs rated by the user monthly (`python -m research rate`);
  judge-human agreement is reported in the paper.

## Splits

- `curated`: hand-written probes with annotations
  (research/data/evalset/curated.yaml), categories: preference_recall, style,
  routine, correction_persistence, generic_control.
- `harvested` / `replay`: real chat prompts under a **temporal split** — an
  artifact trained on data through day N is evaluated only on prompts from day
  N+1 onward. Weekly frozen artifacts are evaluated on the following week's
  prompts. These prompts carry category `harvested`: they are whatever the user
  happened to say, so they are *not* labelled as personalization probes and
  carry no annotations.
- Condition 4 below is therefore measured on the **curated** split only, which
  is the only split containing real `generic_control` probes. The curated split
  runs weekly (`nightly --weekly`, automatic on Sundays); the replay split runs
  nightly.

## Definition of "improved" (the pre-registered bar)

An arm beats base when ALL of:

1. Position-swap-averaged win rate vs base **> 55%**,
2. Exact two-sided sign test (ties excluded) **p < 0.05**,
3. **n ≥ 30** judged comparisons,
4. `generic_control` win rate vs base **≥ 45%** (no regression on
   non-personal ability),
5. No style-compliance regression (arm style score ≥ base style score − 0.05),
6. Conditions 1–5 hold on **two consecutive weekly evals**.

## Reporting

- One row per (arm, date, split) appended to `results/eval.csv`; markdown
  digest per run in `results/nightly/`. Aggregates only — transcripts and
  responses never leave `~/.friday/`.
- Negative results are reported with the same prominence as positive ones;
  a gated (non-advanced) arm-B adapter is a data point, not a failure to hide.

## Changelog

- 2026-07-18: initial version.
- 2026-07-19: base model switched Llama-3.1-8B-4bit → Qwen3-8B-4bit (thinking
  disabled) before any eval or training run; local judge auditor swapped
  qwen3:8b → llama3.1 to keep it a different family from the judged base.
- 2026-08-14: eval-shape clarification, before any result was produced (the
  study had collected no data until the ingress bug below was fixed).
  Replay/harvested prompts are now labelled category `harvested` rather than
  `preference_recall`, which was a fiction — they are unannotated real
  utterances. Consequently `control_win_rate` is left blank for splits with no
  `generic_control` probes instead of recording the neutral 0.500 that
  `win_rate_for` returns for "no data", which would have satisfied condition 4
  for free. Condition 4 is measured on the curated split, which runs weekly.
  The six conditions are now evaluated in code (research/protocol.py) rather
  than by eye. No threshold changed.
- 2026-08-14: fixed the ingress bug that starved the study — free-chat turns
  returned the intent router's draft reply and never reached the LLM provider,
  so no exchange was ever recorded with a context snapshot and no shadow,
  replay or eval could run. All data collection begins after this date.
- 2026-08-22: before any eval run, six of the seven FILL-IN probes (pref-01..04,
  routine-01..02) were annotated with the user's real preferences, and ten
  style/behaviour probes (style-06..15) were added from a written voice spec
  (even-toned bad news and corrections, anticipation over permission-asking,
  composure, information density, one-question handling of ambiguity). The
  curated split is now 32 probes; correct-01 remains FILL-IN until the user
  names a real correction, so the curated split still refuses to run.
- 2026-08-22 (later): correct-01 filled from a real correction made the same
  day (Friday said it could not tell jokes; the user said it should). Prompt
  changed from the milk placeholder to the joke request. The curated split now
  has no FILL-IN probes and runs from the first Sunday eval (2026-08-23).
