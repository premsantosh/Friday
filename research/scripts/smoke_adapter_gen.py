"""M0 smoke test: load the 4-bit base + a LoRA adapter via mlx_lm and generate.

Validates arm B's serving path (nightly batch replay uses mlx_lm directly,
never Ollama). Reports tokens/sec and peak memory.

Usage:
    .venv/bin/python research/scripts/smoke_adapter_gen.py \
        [--adapter ~/.friday/research/smoke/adapter]
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

MODEL = "mlx-community/Qwen3-8B-4bit"

PROMPTS = [
    "What's a good way to brew coffee at home?",
    "Give me a two-sentence fact about lighthouses.",
    "How would I explain chess to a child?",
    "Write a one-line reminder about fermentation.",
    "Summarise the plot of a story about glaciers.",
    "What should I cook tonight if I'm tired?",
    "Explain tides in one paragraph.",
    "Give me a short packing list for a rainy weekend.",
    "What's one habit that improves focus?",
    "Describe typography to someone who has never heard the word.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(Path.home() / ".friday/research/smoke/adapter"))
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()

    adapter = Path(args.adapter).expanduser()
    adapter_path = str(adapter) if adapter.exists() else None
    print(f"model: {MODEL}")
    print(f"adapter: {adapter_path or 'NONE (base only — run smoke_qlora.py first)'}")

    t0 = time.monotonic()
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    print(f"load time: {time.monotonic() - t0:.1f}s")

    total_tokens = 0
    t0 = time.monotonic()
    for i, user_msg in enumerate(PROMPTS):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,  # match study conditions (Qwen3 answers directly)
        )
        text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
        total_tokens += len(tokenizer.encode(text))
        print(f"[{i + 1}/{len(PROMPTS)}] {user_msg!r} -> {len(text)} chars")
    gen_time = time.monotonic() - t0

    peak_gb = mx.get_peak_memory() / 1e9
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print("\n--- smoke_adapter_gen summary ---")
    print(f"completions:   {len(PROMPTS)}")
    print(f"gen tokens/s:  {total_tokens / gen_time:.1f} (approx, includes prompt processing)")
    print(f"mlx peak mem:  {peak_gb:.2f} GB")
    print(f"process RSS:   {rss_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
