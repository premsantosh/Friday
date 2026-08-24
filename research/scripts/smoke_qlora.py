"""M0 smoke test: can this Mac QLoRA-train an 8B-4bit model without swap-thrashing?

Standalone: imports nothing from Friday. Generates a small synthetic chat dataset,
runs `mlx_lm lora` in a subprocess, and samples system memory every 5 s to a CSV.

Usage:
    .venv/bin/python research/scripts/smoke_qlora.py                 # Ollama resident (worst case)
    .venv/bin/python research/scripts/smoke_qlora.py --no-preload    # Ollama unloaded first

Pass criterion: training completes and free memory never enters sustained
swap-thrash (watch min_free_pct and swapout growth in the summary).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODEL = "mlx-community/Qwen3-8B-4bit"
OLLAMA_RESIDENT_MODEL = "qwen3:8b"  # the shadow model — worst-case co-residency
OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42

# Deterministic synthetic chat data: enough variety to make the loss move,
# no personal content.
TOPICS = [
    ("What's a good way to brew {} at home?", "For {}, a consistent grind and water at 93C matter most."),
    ("Summarise the plot of a story about {}.", "It follows {} through a series of small failures toward one success."),
    ("Give me a two-sentence fact about {}.", "{} has a longer history than most assume. Its modern form emerged in the last century."),
    ("How would I explain {} to a child?", "Think of {} like a game with simple rules that combine into something big."),
    ("Write a one-line reminder about {}.", "Don't forget: {} needs attention before the end of the week."),
]
SUBJECTS = [
    "coffee", "sourdough", "tides", "chess", "volcanoes", "jazz", "kites",
    "cartography", "beekeeping", "typography", "orbits", "fermentation",
    "lighthouses", "puzzles", "gardens", "trains", "glaciers", "violins",
    "bridges", "clocks",
]


def make_dataset(data_dir: Path, n_train: int = 100, n_valid: int = 20) -> None:
    rng = random.Random(SEED)
    rows = []
    for _ in range(n_train + n_valid):
        q, a = rng.choice(TOPICS)
        s = rng.choice(SUBJECTS)
        rows.append({"messages": [
            {"role": "user", "content": q.format(s)},
            {"role": "assistant", "content": a.format(s)},
        ]})
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("train", rows[:n_train]), ("valid", rows[n_train:])):
        with open(data_dir / f"{name}.jsonl", "w") as f:
            for r in chunk:
                f.write(json.dumps(r) + "\n")


def ollama_set_residency(load: bool) -> str:
    """Load the shadow model resident (worst case) or unload it. Returns a status string."""
    body = json.dumps({"model": OLLAMA_RESIDENT_MODEL, "prompt": "",
                       "keep_alive": "30m" if load else 0}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        return "loaded resident (keep_alive 30m)" if load else "unloaded (keep_alive 0)"
    except Exception as e:  # Ollama not running is fine for the no-preload case
        return f"ollama unavailable: {e}"


def read_memory() -> dict:
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    free_pct = int(m.group(1)) if (m := re.search(r"free percentage: (\d+)", out)) else -1
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout

    def pages(label: str) -> int:
        mm = re.search(rf"{label}:\s+(\d+)", vm)
        return int(mm.group(1)) if mm else 0

    return {
        "free_pct": free_pct,
        "pages_free": pages("Pages free"),
        "pages_compressed": pages(r"Pages stored in compressor"),
        "swapins": pages("Swapins"),
        "swapouts": pages("Swapouts"),
    }


def sampler(csv_path: Path, stop: threading.Event) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = None
        while not stop.is_set():
            row = {"t": round(time.time(), 1), **read_memory()}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            stop.wait(5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-preload", action="store_true", help="unload Ollama before training")
    ap.add_argument("--out", default=str(Path.home() / ".friday/research/smoke"))
    args = ap.parse_args()

    out = Path(args.out)
    data_dir = out / "data"
    adapter_dir = out / "adapter"
    make_dataset(data_dir)

    status = ollama_set_residency(load=not args.no_preload)
    print(f"ollama {OLLAMA_RESIDENT_MODEL}: {status}")

    csv_path = out / ("mem_preload.csv" if not args.no_preload else "mem_nopreload.csv")
    stop = threading.Event()
    t = threading.Thread(target=sampler, args=(csv_path, stop), daemon=True)
    t.start()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--batch-size", "1",
        "--num-layers", "16",
        "--iters", str(args.iters),
        "--learning-rate", "1e-5",
        "--max-seq-length", "1024",
        "--seed", str(SEED),
        "--adapter-path", str(adapter_dir),
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--mask-prompt",
    ]
    print("running:", " ".join(cmd))
    start = time.monotonic()
    proc = subprocess.run(cmd)
    wall = time.monotonic() - start
    stop.set()
    t.join(timeout=10)

    rows = list(csv.DictReader(open(csv_path)))
    free = [int(r["free_pct"]) for r in rows if int(r["free_pct"]) >= 0]
    swapout_delta = int(rows[-1]["swapouts"]) - int(rows[0]["swapouts"]) if rows else 0
    print("\n--- smoke_qlora summary ---")
    print(f"exit code:      {proc.returncode}")
    print(f"wall time:      {wall:.0f}s for {args.iters} iters")
    print(f"free pct:       start {free[0] if free else '?'}  min {min(free) if free else '?'}")
    print(f"swapout delta:  {swapout_delta} pages")
    print(f"memory csv:     {csv_path}")
    print(f"adapter:        {adapter_dir}")
    verdict = "PASS" if proc.returncode == 0 and (not free or min(free) > 5) else "CHECK"
    print(f"verdict:        {verdict} (inspect csv for sustained swapping)")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
