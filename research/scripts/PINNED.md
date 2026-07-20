# Pinned versions and parameters (research reproducibility)

Recorded 2026-07-18 on the study machine (M4 MacBook, 16 GB unified memory, macOS Darwin 25.3.0).

## Toolchain
- Python (research worktree `.venv`, where the study runs): 3.14.3
  (main checkout's venv is 3.10.10; mlx versions identical in both)
- mlx: 0.32.0
- mlx-lm: 0.31.3
  - Training objectives available: SFT only (`--fine-tune-type lora|dora|full`).
    No DPO/KTO/ORPO in mlx-lm proper as of this version. Phase-2 preference
    tuning will need a third-party package (e.g. mlx-lm-lora) or an SFT-only fallback.
- Ollama models present: llama3.1:latest (8B, 4.9 GB), qwen3:8b, qwen3:4b,
  qwen3:1.7b, llama3.2:3b, gemma3:4b, phi3:mini (extractor, pulled during M0)

## Base model (fixed unless a logged decision changes it)
- **2026-07-19: switched to Qwen3-8B before any eval/training run** (user
  decision; stronger local base). Local stack is now:
  - Shadow + memory-observer (Ollama): `qwen3:8b`, `think: false`
  - Training + replay/eval base (MLX): `mlx-community/Qwen3-8B-4bit`,
    `enable_thinking=False`, `<think>` blocks stripped defensively
  - Local judge auditor: `llama3.1` (different family from the judged base)

## Arm B training (fixed unless a logged decision changes them)
- Base model: `mlx-community/Qwen3-8B-4bit` (QLoRA via 4-bit base)
- LoRA: rank 8 (mlx-lm default), 16 layers, batch size 1, lr 1e-5,
  max seq 1024, seed 42, `--mask-prompt`, iters = min(400, 4 x n_examples)
- Serving: `mlx_lm.load(base, adapter_path=...)` — Ollama is never involved

## Smoke test results on Qwen3-8B-4bit — measured 2026-07-19

### smoke_qlora.py — PASS (worst case: Ollama qwen3:8b held resident)
- 100 iters, batch 1, rank 8, 16 layers, lr 1e-5, seed 42: 1.80 it/s,
  ~42 tok/s trained, MLX peak 5.21 GB; wall 283 s including model download
- Free memory dipped to 8% (qwen3:8b resident is 5.2 GB vs llama's 4.9);
  272k pages swapped, no thrash, full speed throughout. Real nightly runs
  have more headroom: train_nightly always unloads Ollama first.
- Loss 1.30@30 → 0.001@100 — same fast-overfit shape as the Llama run.

### smoke_adapter_gen.py — PASS
- `mlx_lm.load(base, adapter_path=...)` with `enable_thinking=False`:
  3.1 s load, 14.4 tok/s, MLX peak 4.80 GB, RSS 2.73 GB
- Adapter effect visible (trained-register prompts short; the one
  out-of-distribution prompt answers long), no <think> leakage observed.

## Historical: smoke results on Meta-Llama-3.1-8B-Instruct-4bit — measured 2026-07-18

### smoke_qlora.py — PASS (worst case: Ollama llama3.1 held resident)
- 100 iters, batch 1, rank 8 (mlx-lm default), 16 layers, lr 1e-5, seed 42
- Wall: 348 s including first-time 4.3 min model download; training itself
  ~1.6 it/s (~65 s of compute + 3 validation passes)
- MLX peak training memory: 5.38 GB (short synthetic seqs; expect more at
  real conversation lengths, but ample headroom)
- System free memory: never below 22%; swapout delta 254k pages during the
  window (some swapping, no thrash, run completed normally)
- Loss 6.66 → 0.03 by iter 50 on 100 examples — confirms tiny-dataset
  overfitting is fast; replay mixing + few iters are mandatory, not optional
- Implication: nightly 400-iter training ≈ 4-5 min of compute. Arm B is GO.

### smoke_adapter_gen.py — PASS
- `mlx_lm.load(base, adapter_path=...)`: 2.4 s load, 13.7 tok/s generation,
  MLX peak 4.86 GB, process RSS 3.32 GB
- Adapter effect visible: trained-style prompts answer in the short synthetic
  register; the one prompt outside the training distribution answers long.

## Manual live smoke checklist (post-M1)
1. `FRIDAY_RESEARCH=1 python main.py --telegram`
2. Send a chat message from the allowlisted Telegram account; confirm a row in
   `~/.friday/research.db` `exchanges` and a shadow row in `shadow_responses`.
3. Tap 👍 on the reply; confirm a `feedback` row with kind=explicit, signal=+1.
4. `python -m research nightly --dry-run` (skips training, FakeJudge).
