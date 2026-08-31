# Methods: the Friday personalization study

Consolidated methods for the paper. The normative protocol is
`research/data/evalset/PROTOCOL.md` (pre-registered, changelogged); this
document explains the design and names its limitations. Pinned versions and
measured smoke results: `research/scripts/PINNED.md`. Column dictionary:
`results/README.md`.

## Design

Single-subject (N-of-1) longitudinal study on one user, one device (M4 Mac,
16 GB), one persona, one language. Question: does a personalization arm make a
local 8B model's replies better *for this user* than the vanilla local base?
Production (Claude Haiku) answers the user; the study runs in shadow and
overnight, and never touches the reply path.

## Population and sampling bias

The study corpus is free-chat turns that reached the LLM provider: cache hits,
tool-using turns and workflow commands are excluded by construction
(`agent/engine.py` records only `chat` turns; workflow routes carry no context
snapshot). The evaluated population is therefore "cache-missing, tool-free
chat", a non-random subset of real usage; the selection rate is not recorded.

## Arms

- `base` — pinned `mlx-community/Qwen3-8B-4bit`, thinking disabled, static
  persona prompt, greedy decoding. Opponent for every comparison.
- `facts` — production's keyword facts store injected as a system block.
  **A deliberately moving baseline**: it reads the live `~/.friday/memory.db`
  at generation time and is not versioned. Production retrieval changed on
  2026-08-30 (see changelog); since then the replay stage records a sha256 of
  each injected block (`replay.context` events) so drift is visible, though
  not reproducible.
- `memory` (arm A) — nightly reflection memory, versioned snapshots.
- `lora` (arm B) — nightly QLoRA of the base (rank 8, seed 42, full retrain
  each night, style-canary promotion gate), versioned adapters with manifests.
- `prompt` (arm C) — Sonnet-evolved preference block, versioned. **No quality
  gate**: `current` advances unconditionally.

## Generation control

All arms generate through one runtime (mlx_lm), one base, greedy decoding,
max 200 tokens, the identical stored context snapshot per exchange; the arm's
artifact is the only delta (`research/generate.py`).

## Judging

Primary: `claude-sonnet-5`, pairwise, both presentation orders averaged
(win 1 / tie 0.5 / loss 0), judge errors scored as ties, temperature 0,
versioned rubric (`r1:<sha8>`, pinned by test). Audit: llama3.1 (different
family from the judged base) re-judges a deterministic 20% sample; direction
agreement recorded per row, blank when unavailable. Human anchor: ~20
anonymized arm-vs-base pairs weekly, seeded presentation order; judge–human
direction agreement reported.

## Splits and the prospective cutoff

- `curated` (weekly): hand-written probes with annotations; the only split
  with `generic_control` probes. A held-out `reserve` subset is evaluated only
  once, for the final paper numbers.
- `replay` (nightly): real chat prompts. Stage order is
  harvest → replay → eval → learners → report → publish, so each night judges
  the artifacts built the previous night; additionally, replay only evaluates
  exchanges newer than the latest learner-consumption event. Replay rows dated
  ≤ 2026-08-30 predate this fix and are non-evidential.

## Endpoints

Primary (amended 2026-08-30, pre-registered before any pooled analysis):
pooled curated verdicts over 4 consecutive weekly evals per arm; pooled
decisive win rate with Wilson 95% CI; exact two-sided sign test on pooled
decisive pairs; Holm correction across the 3 study arms; pooled control ≥ 45%
and style delta > −0.05. Secondary: the original per-week six-condition bar,
retained descriptively.

Power (exact sign test, α = .05): the per-week bar at n = 32 has power 0.041
at a true decisive-win probability of 0.55 (its own threshold) and needs
~0.77 for 80% power; the pooled design reaches 80% power near 0.64 with ~108
decisive comparisons.

## Provenance

Every store write emits an event in the same transaction (enforced by test);
artifacts carry manifests naming exact input exchange ids, dataset hashes,
training params (incl. LoRA rank, mlx/mlx-lm versions) and git rev; every
eval row records git rev, rubric id and judge temperature; context snapshots
record their format version, ingress engine and production decode params
(from 2026-08-30, `snapshot_format: 2`). `python -m research trace` walks any
exchange, artifact or run.

## Limitations (disclosed, not fixed)

- **N-of-1**: one user, persona, language, device. Findings are a case study,
  not a generalization claim.
- **Judge-as-synthesizer entanglement**: Sonnet synthesizes arm B's correction
  targets and writes arm C's preference block, and also judges — a shared-
  preference confound on two of three arms.
- **Style proxy is two rules** (no exclamation marks, ≤ 4 sentences);
  `addresses_user` is computed but unused. `arm_style` measures less than the
  name suggests.
- **Facts arm is unreproducible** by design (live store); drift is hashed,
  not captured.
- **Miner precision unvalidated**: implicit feedback (rephrase/correction/
  thanks) comes from string heuristics with no labeled sample; a false
  `thanks` admits an exchange into LoRA training immediately.
- **Snapshot heterogeneity**: provider-path snapshots are captured, engine-path
  snapshots are rebuilt; formats before 2026-08-30 carry no version marker.
- **Pre-fix data**: replay rows ≤ 2026-08-30 are contaminated (excluded);
  Sonnet verdicts before 2026-08-30 were produced at API-default temperature
  under rubric r1's text (unversioned at the time).
- **Eval-set reuse**: the same curated probes run weekly; the reserve split
  mitigates but does not eliminate drift toward the probes.
- The LoRA arm's 1:1 Dolly replay mix pairs the persona prompt with
  off-persona targets — a plausible mechanism for its observed style
  regression; isolating it is future work (no ablation run).
