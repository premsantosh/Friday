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
  (win 1, tie 0.5, loss 0). Judge errors score as ties. Temperature pinned
  to 0; the rubric is versioned (research/judge.py `rubric_id()`), and both
  are recorded on every eval row.
- Agreement audit: llama3.1 (a different family from the judged Qwen3 base,
  keeping the audit independent) re-judges a deterministic 20% sample each
  eval; direction-agreement on shared decisive prompts is recorded per row
  (`local_agreement`, `n_audited`; blank when Ollama is unavailable — never
  imputed).
- Human anchor: ~20 arm-vs-base pairs from the most recent eval run rated by
  the user weekly (`python -m research rate --mode arm-base`), anonymized with
  a seeded presentation order; judge–human direction agreement is reported in
  the digest and the paper.

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

## Primary endpoint (amended 2026-08-30, see changelog)

Per study arm (prompt, memory, lora; `facts` is a baseline, `base` the
opponent):

- Pool the per-prompt curated verdicts from **4 consecutive weekly evals** of
  one artifact lineage (target ~144 comparisons; ~108 decisive at observed tie
  rates; 80% power at a true decisive-win probability of ~0.64).
- Report the pooled decisive win rate with a **Wilson 95% CI** and the exact
  two-sided sign test on pooled decisive pairs.
- **Holm correction** across the 3 study arms. An arm beats base when its
  Holm-adjusted p < 0.05 AND pooled `generic_control` win rate >= 45% AND
  pooled style delta > -0.05.
- Replay rows pool separately as a supporting (real-utterance) outcome,
  prospective rows only (date >= 2026-08-31).

Implemented in `research/protocol.py::evaluate_pooled` / `holm` /
`wilson_interval`; `python -m research protocol --pooled` prints it.

## Definition of "improved" (the per-week bar; retained as a descriptive secondary outcome)

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
- 2026-08-30: production memory retrieval changed (word-split fact search,
  confidence floors at store and retrieval, identity-placeholder block appended
  to prompts). This alters the `facts` arm's inputs and the content of every
  future context snapshot mid-study. `facts` is a moving baseline by design;
  from the stage-reorder date the replay stage records a sha256 of the injected
  facts block per exchange (`replay.context` events) so drift is visible.
- 2026-08-30: nightly stage order changed to harvest -> replay -> eval ->
  reflect -> evolve -> train -> report -> publish, plus a prospective cutoff:
  replay only evaluates exchanges newer than the latest learner-consumption
  event (memory.consumed/observed, prompt.consumed, dataset.built). Every
  `split=replay` row dated on or before 2026-08-30 was produced under the old
  order (artifacts trained the same night on the judged exchanges) and is
  NON-EVIDENTIAL — see results/README.md. Curated rows are unaffected (probes
  never enter training data). Prospective replay rows carry notes=prospective;
  pooled analyses use `protocol.REPLAY_PROSPECTIVE_FROM = 2026-08-31`.
- 2026-08-30: judge-agreement audit wired as pre-registered (it had never run):
  llama3.1 re-judges a deterministic 20% sample per (date, split, arm);
  agreement and n recorded per eval row, blank when Ollama is down. Audit
  verdicts are persisted under judge `local:llama3.1`.
- 2026-08-30: SonnetJudge pinned to temperature=0 and the rubric versioned
  (r1, sha-pinned by tests/test_judge_rubric.py); git_rev, rubric id and judge
  temperature recorded on every eval row and in eval_results.scores. Verdicts
  before this date were produced at the API-default temperature.
- 2026-08-30: human anchor changed from monthly production-vs-shadow to weekly
  ~20 arm-vs-base pairs from the latest eval run (replay via shadow_responses,
  curated via the new curated_responses table), anonymized, seeded order; a
  Telegram nudge follows each weekly eval. Curated-split generations are now
  persisted (previously discarded after judging).
- 2026-08-30: amended PRIMARY endpoint, pre-registered before any pooled
  analysis was run. Power analysis (exact two-sided sign test, alpha=.05): the
  per-week bar at n=32 has power 0.041 at a true decisive-win probability of
  0.55 and reaches 80% power only around 0.77 — it cannot detect its own
  threshold. New primary: pooled curated verdicts over 4 consecutive weekly
  evals (~108 decisive; 80% power near 0.64), Wilson 95% CI, exact sign test,
  Holm correction across the 3 study arms, plus the pooled control and style
  conditions (see "Primary endpoint" above). The per-week bar is retained
  unchanged as a descriptive secondary outcome. Tie-inclusive win rates
  continue to be reported alongside.
- 2026-08-30: curated set restructured for the amended endpoint: +12 draft
  probe skeletons (excluded from every load until the user writes real
  annotations; exempt from the FILL-IN refusal) targeting 44 total, and a
  held-out reserve of 8 probes (`reserve: true`, style-15 and control-10 moved
  there now, 6 more as drafts are filled) evaluated only via
  `research eval --split reserve` for the final paper numbers. Weekly n is 30
  until drafts are written, then 36. Weekly judge cost rises ~12.5% once full.
- 2026-08-30: nightly gains a publish stage: results/eval.csv and
  results/nightly/*.md are auto-committed (never pushed) after a PII scan
  (emails, phone-like numbers, a user-configured name list via
  FRIDAY_PII_NAMES or ~/.friday/pii_names.txt); findings block the commit and
  alert via Telegram. eval.csv gained columns git_rev, rubric,
  judge_temperature, local_agreement, n_audited, wilson_low, wilson_high,
  notes (older rows are shorter; readers treat missing fields as blank).
