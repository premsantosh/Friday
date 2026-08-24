"""Build the pinned replay-mixing dataset (one-time, committed to the repo).

Samples 500 general instruction examples from databricks-dolly-15k (CC-BY-SA
3.0), seeded, converted to chat format. Mixed 1:1 with personal data during
arm-B training so nightly fine-tunes don't forget general ability.

Usage: .venv/bin/python research/scripts/build_replay.py
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
       "resolve/main/databricks-dolly-15k.jsonl")
OUT = Path(__file__).parent.parent / "data" / "replay" / "replay.jsonl"
N = 500
SEED = 42
MAX_CHARS = 1500  # keep sequences short; training is batch=1 on 16 GB


def main() -> int:
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    # Short, context-free examples only: replay is about retaining general
    # instruction-following, not long-document QA.
    eligible = [
        r for r in rows
        if not r.get("context")
        and len(r["instruction"]) + len(r["response"]) <= MAX_CHARS
    ]
    sample = random.Random(SEED).sample(eligible, N)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in sample:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["response"]},
            ]}) + "\n")
    print(f"wrote {N} examples to {OUT} "
          f"(from {len(eligible)} eligible of {len(rows)} total, seed {SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
