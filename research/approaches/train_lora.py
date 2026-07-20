"""Arm B training orchestration: guard memory, retrain from scratch on the
full growing dataset, gate on the forgetting canary, version the adapter.

Never stacks adapters (nightly sequential fine-tuning compounds drift); every
run trains fresh from the pinned 4-bit base on everything collected so far.
A failed quality gate keeps `current` where it was and logs the failure — for
this study a gated adapter is a data point about tiny-dataset stability, not
an error to retry.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from research import artifacts
from research.approaches import lora_pipeline
from research.db import ResearchStore

logger = logging.getLogger(__name__)

ARM = "lora"
BASE_MODEL = "mlx-community/Qwen3-8B-4bit"  # must match research/generate.py BASE_MODEL
SEED = 42
LEARNING_RATE = "1e-5"
NUM_LAYERS = 16
MAX_SEQ = 1024
MAX_ITERS = 400
MIN_EXAMPLES = 20          # below this a fine-tune is pure overfitting
MIN_FREE_PCT = 15          # measured: training passed with 22% free
GATE_MIN_STYLE = 0.6       # forgetting canary threshold on generic_control
OLLAMA_URL = "http://localhost:11434"


def free_memory_pct() -> int:
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                         text=True).stdout
    m = re.search(r"free percentage: (\d+)", out)
    return int(m.group(1)) if m else -1


def unload_ollama(models: tuple[str, ...] = ("llama3.1", "qwen3:8b")) -> None:
    """Ask Ollama to release resident models before training. Best-effort."""
    import requests
    for model in models:
        try:
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": model, "prompt": "", "keep_alive": 0},
                          timeout=30)
        except Exception:
            pass


def run_training(data_dir: Path, adapter_dir: Path, n_examples: int) -> int:
    iters = min(MAX_ITERS, 4 * n_examples)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--batch-size", "1",
        "--num-layers", str(NUM_LAYERS),
        "--iters", str(iters),
        "--learning-rate", LEARNING_RATE,
        "--max-seq-length", str(MAX_SEQ),
        "--seed", str(SEED),
        "--adapter-path", str(adapter_dir),
        "--steps-per-report", "50",
        "--steps-per-eval", "100",
        "--mask-prompt",
    ]
    logger.info("training: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (adapter_dir / "train.log").write_text(proc.stdout[-20000:] + proc.stderr[-20000:])
    return proc.returncode


def quality_gate(adapter_path: Path) -> tuple[bool, str]:
    """Forgetting canary: generic_control probes must stay in persona and
    non-empty with the new adapter. Deterministic checks only — free."""
    from research.evalset import load_curated
    from research.generate import ArmSpec, generate_candidates
    from research.style import style_score

    controls = [p for p in load_curated() if p.category == "generic_control"]
    responses = generate_candidates(
        controls, ArmSpec(name=ARM, adapter_path=str(adapter_path)))
    texts = [t for t in responses.values() if t]
    if len(texts) < len(controls) * 0.8:
        return False, f"only {len(texts)}/{len(controls)} controls generated"
    score = style_score(texts)
    if score < GATE_MIN_STYLE:
        return False, f"style collapse: {score:.2f} < {GATE_MIN_STYLE}"
    return True, f"style {score:.2f} on {len(texts)} controls"


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def train_nightly(
    store: ResearchStore,
    date_str: str,
    *,
    artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR,
    correction_llm_fn=None,
) -> str:
    """One nightly training run. Returns a status string for the runs table."""
    free = free_memory_pct()
    if 0 <= free < MIN_FREE_PCT:
        unload_ollama()
        time.sleep(5)
        free = free_memory_pct()
        if 0 <= free < MIN_FREE_PCT:
            return f"skipped: only {free}% memory free after unloading Ollama"
    else:
        unload_ollama()

    version_dir = artifacts.new_version(ARM, date_str, Path(artifacts_dir))
    data_dir = version_dir / "data"
    corrections_cache = artifacts.arm_dir(ARM, Path(artifacts_dir)) / "corrections.jsonl"
    stats = lora_pipeline.build_dataset(
        store, data_dir,
        correction_llm_fn=correction_llm_fn,
        corrections_cache=corrections_cache,
    )
    if stats["n_personal"] < MIN_EXAMPLES:
        return (f"skipped: {stats['n_personal']} personal examples "
                f"< {MIN_EXAMPLES} minimum (adapter dir {version_dir.name} left without weights)")

    code = run_training(data_dir, version_dir, stats["n_train"])
    if code != 0:
        return f"FAILED: mlx_lm lora exit {code} (see {version_dir}/train.log)"

    (version_dir / "config.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "seed": SEED, "learning_rate": LEARNING_RATE,
        "num_layers": NUM_LAYERS, "max_seq": MAX_SEQ,
        "iters": min(MAX_ITERS, 4 * stats["n_train"]),
        "git_rev": git_rev(), **stats,
    }, indent=1))

    passed, gate_note = quality_gate(version_dir)
    if not passed:
        (version_dir / "GATED").write_text(gate_note + "\n")
        return f"trained but GATED (current unchanged): {gate_note}"
    artifacts.advance_current(ARM, version_dir.name, Path(artifacts_dir))
    return f"advanced to {version_dir.name} ({stats['n_train']} train ex): {gate_note}"
