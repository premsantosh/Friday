# Results

Longitudinal record of the personalization study. Aggregates only — transcripts
and responses never leave `~/.friday/`. The protocol lives in
`research/data/evalset/PROTOCOL.md`; methods in `docs/research-methods.md`.

## Files

- `eval.csv` — one row per (arm, date, split), append-only. Written by the
  nightly report stage; committed by the publish stage after a PII scan.
- `nightly/YYYYMMDD.md` — per-run digest (tables, per-category rates, the
  pre-registered bar, the pooled primary endpoint, human-anchor lines).
- `paper/` — output of `python -m research paper` (tables, figures).

## Data-quality note: replay rows dated on or before 2026-08-30

The nightly loop originally ran train-then-replay, so `split=replay` rows up to
and including 2026-08-30 evaluated arms on the same exchanges they were built
from that night. Those rows are **non-evidential** and are excluded from the
pooled primary endpoint (`protocol.REPLAY_PROSPECTIVE_FROM`). Replay rows from
2026-08-31 onward carry `notes=prospective` and are produced under the fixed
order (replay/eval before any learner) plus a learner-consumption cutoff.
Curated rows are unaffected: probes never enter training data.

## eval.csv columns

| column | meaning |
|---|---|
| date, run_id, arm, opponent | which comparison, which nightly run |
| artifact_version | the arm's artifact **as judged** (captured at replay time) |
| judge | `sonnet:*` primary, `local:*` audit, `human` anchor, `fake` dry-run (never evidence) |
| split | `replay` (nightly, real prompts) / `curated` (weekly probes) |
| n_prompts, n_decisive, wins, losses | position-swap-averaged outcomes |
| win_rate | tie-inclusive mean score |
| p_value | exact two-sided sign test, ties excluded |
| n_control, control_win_rate | generic_control probes (blank when none — never imputed) |
| arm_style, opponent_style | deterministic style-proxy scores |
| git_rev, rubric, judge_temperature | provenance of the code and instrument |
| local_agreement, n_audited | llama3.1 audit (blank when Ollama was down) |
| wilson_low, wilson_high | Wilson 95% CI on decisive wins (blank at 0 decisive) |
| notes | `prospective` on post-fix replay rows |

Rows written before 2026-08-30 are shorter (the header was extended in place);
readers treat the missing fields as blank.
