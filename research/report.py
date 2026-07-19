"""Results emitters: results/eval.csv (append-only) + per-run markdown.

Only aggregates leave ~/.friday — never transcripts or responses. The CSV is
the longitudinal record the paper's plots come from; the markdown is the
human-readable nightly digest.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from research.eval_runner import PairwiseResult

RESULTS_DIR = Path(__file__).parent.parent / "results"

CSV_FIELDS = [
    "date", "arm", "opponent", "artifact_version", "judge", "split",
    "n_prompts", "n_decisive", "wins", "losses", "win_rate", "p_value",
    "control_win_rate", "arm_style", "opponent_style",
]


def append_csv_row(result: PairwiseResult, *, split: str,
                   artifact_version: str = "", date: str = "",
                   results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "eval.csv"
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "arm": result.arm,
            "opponent": result.opponent,
            "artifact_version": artifact_version,
            "judge": result.judge,
            "split": split,
            "n_prompts": len(result.outcomes),
            "n_decisive": result.n_decisive,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": f"{result.win_rate:.3f}",
            "p_value": f"{result.p_value:.4f}",
            "control_win_rate": f"{result.win_rate_for('generic_control'):.3f}",
            "arm_style": f"{result.arm_style:.3f}",
            "opponent_style": f"{result.opponent_style:.3f}",
        })
    return path


def write_run_markdown(results: list[PairwiseResult], *, split: str,
                       date: str = "", notes: str = "",
                       results_dir: Path = RESULTS_DIR) -> Path:
    date = date or datetime.now().strftime("%Y-%m-%d")
    out_dir = results_dir / "nightly"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date}.md"

    lines = [
        f"# Eval report — {date}",
        "",
        f"Split: {split}",
        "",
        "| arm | vs | judge | n | win rate | p | control | style (arm/opp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.arm} | {r.opponent} | {r.judge} | {len(r.outcomes)} "
            f"| {r.win_rate:.1%} | {r.p_value:.3f} "
            f"| {r.win_rate_for('generic_control'):.1%} "
            f"| {r.arm_style:.0%}/{r.opponent_style:.0%} |"
        )
    lines += ["", "Per-category win rates:", ""]
    for r in results:
        cats = sorted({o.category for o in r.outcomes})
        detail = ", ".join(f"{c}: {r.win_rate_for(c):.1%}" for c in cats)
        lines.append(f"- **{r.arm}**: {detail}")
    if notes:
        lines += ["", notes]
    lines.append("")
    path.write_text("\n".join(lines))
    return path
