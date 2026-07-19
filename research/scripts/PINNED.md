# Pinned versions and parameters (research reproducibility)

Recorded 2026-07-18 on the study machine (M4 MacBook, 16 GB unified memory, macOS Darwin 25.3.0).

## Toolchain
- Python (project venv `.venv`): 3.10.10
- mlx: 0.32.0
- mlx-lm: 0.31.3
  - Training objectives available: SFT only (`--fine-tune-type lora|dora|full`).
    No DPO/KTO/ORPO in mlx-lm proper as of this version. Phase-2 preference
    tuning will need a third-party package (e.g. mlx-lm-lora) or an SFT-only fallback.
- Ollama models present: llama3.1:latest (8B, 4.9 GB), qwen3:8b, qwen3:4b,
  qwen3:1.7b, llama3.2:3b, gemma3:4b, phi3:mini (extractor, pulled during M0)

## Arm B training (fixed unless a logged decision changes them)
- Base model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (QLoRA via 4-bit base)
- LoRA: rank 8 (mlx-lm default), 16 layers, batch size 1, lr 1e-5,
  max seq 1024, seed 42, `--mask-prompt`, iters = min(400, 4 x n_examples)
- Serving: `mlx_lm.load(base, adapter_path=...)` — Ollama is never involved

## Smoke test results (M0) — measured 2026-07-18

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
