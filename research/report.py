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
    "date", "run_id", "arm", "opponent", "artifact_version", "judge", "split",
    "n_prompts", "n_decisive", "wins", "losses", "win_rate", "p_value",
    "n_control", "control_win_rate", "arm_style", "opponent_style",
    # Appended 2026-08 (provenance + audit + CI); rows before then are shorter
    # and readers treat the missing fields as blank.
    "git_rev", "rubric", "judge_temperature", "local_agreement", "n_audited",
    "wilson_low", "wilson_high", "notes",
]

CONTROL_CATEGORY = "generic_control"


def _ensure_header(path: Path) -> None:
    """Upgrade an older eval.csv header in place when columns were appended.

    Only the header line is rewritten, and only when it is a strict prefix of
    CSV_FIELDS — data rows stay byte-identical (append-only record).
    """
    if not path.exists():
        return
    raw = path.read_bytes()
    head, sep, rest = raw.partition(b"\n")
    existing = next(csv.reader([head.decode()]))
    if existing == CSV_FIELDS:
        return
    if existing != CSV_FIELDS[:len(existing)]:
        raise ValueError(
            f"eval.csv header {existing} is not a prefix of CSV_FIELDS; "
            "refusing to touch the historical record")
    path.write_bytes(",".join(CSV_FIELDS).encode() + sep + rest)


def append_csv_row(result: PairwiseResult, *, split: str,
                   artifact_version: str = "", date: str = "",
                   run_id: int | None = None,
                   extra: dict | None = None,
                   results_dir: Path | None = None) -> Path:
    # Resolved at call time, not bound as a default, so tests can redirect it.
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "eval.csv"
    new_file = not path.exists()
    _ensure_header(path)
    unknown = set(extra or ()) - set(CSV_FIELDS)
    if unknown:
        raise ValueError(f"unknown eval.csv columns: {sorted(unknown)}")
    # A split with no control probes has no control result. Writing the neutral
    # 0.500 that win_rate_for returns for "no data" would read as a measurement
    # and silently satisfy the protocol's no-regression condition.
    n_control = sum(1 for o in result.outcomes if o.category == CONTROL_CATEGORY)
    from research.protocol import wilson_interval
    if result.n_decisive:
        lo, hi = wilson_interval(result.wins, result.n_decisive)
        wilson_low, wilson_high = f"{lo:.3f}", f"{hi:.3f}"
    else:
        wilson_low = wilson_high = ""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            **{
                "wilson_low": wilson_low,
                "wilson_high": wilson_high,
            },
            **(extra or {}),
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "run_id": run_id if run_id is not None else "",
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
            "n_control": n_control,
            "control_win_rate": (f"{result.win_rate_for(CONTROL_CATEGORY):.3f}"
                                 if n_control else ""),
            "arm_style": f"{result.arm_style:.3f}",
            "opponent_style": f"{result.opponent_style:.3f}",
        })
    return path


def write_run_markdown(results: list[PairwiseResult], *, split: str,
                       date: str = "", notes: str = "",
                       protocol_for=None, pooled_for=None,
                       results_dir: Path | None = None) -> Path:
    """Human-readable digest. `protocol_for(arm) -> str` adds the bar verdict;
    `pooled_for(arm) -> str` adds the pooled-primary-endpoint line."""
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    date = date or datetime.now().strftime("%Y-%m-%d")
    out_dir = results_dir / "nightly"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date}.md"

    lines = []
    if path.exists():
        # Two splits can report on the same night; append rather than clobber.
        lines = path.read_text().rstrip().splitlines() + [""]
    else:
        lines = [f"# Eval report — {date}", ""]

    lines += [
        f"## Split: {split}",
        "",
        "| arm | vs | judge | n | win rate | p | control | style (arm/opp) | audit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        n_control = sum(1 for o in r.outcomes if o.category == CONTROL_CATEGORY)
        control = f"{r.win_rate_for(CONTROL_CATEGORY):.1%}" if n_control else "n/a"
        audit = (f"{r.local_agreement:.0%} (n={r.n_audited})"
                 if r.local_agreement is not None else "—")
        lines.append(
            f"| {r.arm} | {r.opponent} | {r.judge} | {len(r.outcomes)} "
            f"| {r.win_rate:.1%} | {r.p_value:.3f} | {control} "
            f"| {r.arm_style:.0%}/{r.opponent_style:.0%} | {audit} |"
        )
    lines += ["", "Per-category win rates:", ""]
    for r in results:
        cats = sorted({o.category for o in r.outcomes})
        detail = ", ".join(f"{c}: {r.win_rate_for(c):.1%}" for c in cats)
        lines.append(f"- **{r.arm}**: {detail}")

    if protocol_for is not None:
        verdicts = [(r.arm, protocol_for(r.arm)) for r in results]
        verdicts = [(arm, v) for arm, v in verdicts if v]
        if verdicts:
            lines += ["", "Pre-registered bar (secondary; see data/evalset/PROTOCOL.md):", ""]
            lines += [f"- **{arm}**: {v}" for arm, v in verdicts]

    if pooled_for is not None:
        pooled = [(r.arm, pooled_for(r.arm)) for r in results]
        pooled = [(arm, v) for arm, v in pooled if v]
        if pooled:
            lines += ["", "Pooled primary endpoint (last 4 weekly evals):", ""]
            lines += [f"- **{arm}**: {v}" for arm, v in pooled]
    if notes:
        lines += ["", notes]
    lines.append("")
    path.write_text("\n".join(lines))
    return path
