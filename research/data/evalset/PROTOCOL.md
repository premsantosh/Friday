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
- `harvested`: real chat prompts under a **temporal split** — an artifact
  trained on data through day N is evaluated only on prompts from day N+1
  onward. Weekly frozen artifacts are evaluated on the following week's
  prompts.

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
