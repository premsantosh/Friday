"""`python -m research paper` — paper-ready tables and figures.

Everything here derives from aggregates (results/eval.csv, the PROTOCOL.md
changelog) plus per-prompt verdict rows in research.db (ids and scores, no
text). Every emitted file passes the same PII gate as the nightly publish
stage; on findings, nothing is written home.

Figures need matplotlib (dev-only, guarded); the tables work without it.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from research import protocol
from research.report import RESULTS_DIR

logger = logging.getLogger(__name__)

PROTOCOL_PATH = Path(__file__).parent / "data" / "evalset" / "PROTOCOL.md"
_CHANGELOG_DATE_RE = re.compile(r"^- (20\d\d-\d\d-\d\d)(?: \(later\))?: (.+)")

ARMS = ("facts", "prompt", "memory", "lora")


def _fmt(x, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x:.1%}" if pct else f"{x:.4f}"


def pooled_table(rows: list[dict]) -> tuple[str, list[dict]]:
    """(markdown, records) for the pooled primary endpoint per arm/split."""
    records = []
    for split in ("curated", "replay"):
        pooled = {arm: protocol.evaluate_pooled(rows, arm=arm, split=split)
                  for arm in ARMS}
        adjusted = protocol.holm({a: p.p_value for a, p in pooled.items()
                                  if a in protocol.STUDY_ARMS_FOR_HOLM and p.dates})
        for arm, p in pooled.items():
            if not p.dates:
                continue
            records.append({
                "arm": arm, "split": split, "evals_pooled": len(p.dates),
                "n": p.n_prompts, "n_decisive": p.n_decisive,
                "wins": p.wins, "losses": p.losses,
                "win_rate": round(p.win_rate, 4),
                "decisive_win_rate": round(p.decisive_win_rate, 4),
                "wilson_low": round(p.wilson_low, 4),
                "wilson_high": round(p.wilson_high, 4),
                "p_value": round(p.p_value, 4),
                "holm_p": round(adjusted[arm], 4) if arm in adjusted else "",
                "control_win_rate": (round(p.control_win_rate, 4)
                                     if p.control_win_rate is not None else ""),
                "style_delta": (round(p.style_delta, 4)
                                if p.style_delta is not None else ""),
            })
    lines = [
        "# Pooled primary endpoint",
        "",
        f"Pooled over the last {protocol.POOL_WINDOW} weekly evals per arm"
        f" (replay: prospective rows since {protocol.REPLAY_PROSPECTIVE_FROM} only)."
        " Holm correction across the study arms"
        f" ({', '.join(protocol.STUDY_ARMS_FOR_HOLM)}); `facts` is a baseline.",
        "",
        "| arm | split | evals | n | decisive | win rate | decisive rate"
        " | Wilson 95% CI | p | Holm p | control | style Δ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        ci = f"[{r['wilson_low']:.1%}, {r['wilson_high']:.1%}]" if r["n_decisive"] else "—"
        lines.append(
            f"| {r['arm']} | {r['split']} | {r['evals_pooled']} | {r['n']}"
            f" | {r['n_decisive']} | {r['win_rate']:.1%} | {r['decisive_win_rate']:.1%}"
            f" | {ci} | {r['p_value']:.4f} | {r['holm_p'] or '—'}"
            f" | {_fmt(r['control_win_rate'] if r['control_win_rate'] != '' else None)}"
            f" | {r['style_delta'] if r['style_delta'] != '' else '—'} |")
    return "\n".join(lines) + "\n", records


def agreement_table(rows: list[dict], store=None) -> str:
    lines = ["# Judge agreement", "", "## Local auditor (llama3.1, 20% sample)", "",
             "| date | split | arm | agreement | n audited |", "|---|---|---|---|---|"]
    any_local = False
    for r in rows:
        rate = (r.get("local_agreement") or "").strip()
        if not rate:
            continue
        any_local = True
        lines.append(f"| {r['date']} | {r['split']} | {r['arm']}"
                     f" | {float(rate):.1%} | {r.get('n_audited') or ''} |")
    if not any_local:
        lines.append("| — | — | — | no audited rows yet | — |")

    lines += ["", "## Human anchor (arm vs base)", ""]
    if store is None:
        lines.append("(research.db not available)")
        return "\n".join(lines) + "\n"
    human = store.query(
        "SELECT run_id, arm, prompt_id, winner FROM eval_results"
        " WHERE judge = 'human' AND opponent = 'base' AND run_id IS NOT NULL")
    if not human:
        lines.append("No human ratings yet — run `python -m research rate --mode arm-base`.")
        return "\n".join(lines) + "\n"
    sonnet = {}
    for r in store.query(
            "SELECT run_id, arm, prompt_id, winner FROM eval_results"
            " WHERE judge LIKE 'sonnet%'"):
        sonnet[(r["run_id"], r["arm"], r["prompt_id"])] = r["winner"]
    per_run: dict = defaultdict(lambda: [0, 0])  # run -> [agree, decisive]
    for r in human:
        s = sonnet.get((r["run_id"], r["arm"], r["prompt_id"]))
        if r["winner"] != "tie" and s in ("arm", "base"):
            per_run[r["run_id"]][1] += 1
            per_run[r["run_id"]][0] += r["winner"] == s
    lines += ["| run | rated | decisive overlap | agreement |", "|---|---|---|---|"]
    counts = defaultdict(int)
    for r in human:
        counts[r["run_id"]] += 1
    for run_id in sorted(per_run):
        agree, decisive = per_run[run_id]
        rate = f"{agree / decisive:.1%}" if decisive else "—"
        lines.append(f"| {run_id} | {counts[run_id]} | {decisive} | {rate} |")
    return "\n".join(lines) + "\n"


def timeline_table(rows: list[dict]) -> str:
    lines = ["# Study timeline", "", "## Protocol changelog", ""]
    try:
        in_changelog = False
        for line in PROTOCOL_PATH.read_text().splitlines():
            if line.startswith("## Changelog"):
                in_changelog = True
                continue
            if in_changelog:
                m = _CHANGELOG_DATE_RE.match(line)
                if m:
                    lines.append(f"- **{m.group(1)}** — {m.group(2)}")
    except OSError:
        lines.append("(PROTOCOL.md not readable)")
    lines += ["", "## Artifact versions per eval", "",
              "| date | split | " + " | ".join(ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    by_date: dict = defaultdict(dict)
    for r in rows:
        if r.get("judge") in protocol.EXCLUDED_JUDGES:
            continue
        by_date[(r["date"], r["split"])][r["arm"]] = r.get("artifact_version") or "—"
    for (date, split), arms in sorted(by_date.items()):
        lines.append(f"| {date} | {split} | "
                     + " | ".join(arms.get(a, "—") for a in ARMS) + " |")
    return "\n".join(lines) + "\n"


def _series(rows: list[dict], split: str):
    """arm -> [(date, win_rate, n)] for real-judge rows of one split."""
    series = defaultdict(list)
    cutoff = protocol.REPLAY_PROSPECTIVE_FROM.replace("-", "")
    for r in rows:
        if r.get("split") != split or r.get("judge") in protocol.EXCLUDED_JUDGES:
            continue
        if split == "replay" and (r.get("date") or "").replace("-", "") < cutoff:
            continue
        wr = (r.get("win_rate") or "").strip()
        if not wr:
            continue
        series[r["arm"]].append((r["date"], float(wr), int(r.get("n_prompts") or 0)))
    return series


def write_figures(rows: list[dict], out_dir: Path, store=None) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures "
              "(pip install matplotlib; tables were still written)")
        return []

    written = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, points in sorted(_series(rows, "curated").items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", label=arm)
        for d, wr, n in points:
            ax.annotate(f"n={n}", (d, wr), textcoords="offset points",
                        xytext=(0, 6), fontsize=7)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("win rate vs base (ties = 0.5)")
    ax.set_title("Curated split, weekly")
    ax.legend()
    fig.autofmt_xdate()
    for ext in ("svg", "pdf"):
        p = out_dir / f"winrate_over_time.{ext}"
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)

    # Per-category decisive win rates from per-prompt verdicts (sonnet only).
    if store is not None:
        cats = _category_scores(store)
        if cats:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            categories = sorted({c for arm in cats.values() for c in arm})
            width = 0.8 / max(1, len(cats))
            for i, (arm, per_cat) in enumerate(sorted(cats.items())):
                xs = [j + i * width for j in range(len(categories))]
                ys = [per_cat.get(c, (0, 0))[0] / per_cat[c][1]
                      if per_cat.get(c, (0, 0))[1] else 0 for c in categories]
                ax.bar(xs, ys, width=width, label=arm)
            ax.set_xticks([j + 0.4 - width / 2 for j in range(len(categories))])
            ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
            ax.set_ylabel("decisive win rate vs base")
            ax.set_title("Per-category (sonnet verdicts, prospective replay only)")
            ax.legend()
            for ext in ("svg", "pdf"):
                p = out_dir / f"category_breakdown.{ext}"
                fig.savefig(p, bbox_inches="tight")
                written.append(p)
            plt.close(fig)
    return written


def _category_scores(store) -> dict:
    """arm -> {category: (wins, decisive)} from judge.verdict-backed rows."""
    from research.evalset import load_curated

    try:
        cat_by_id = {p.id: p.category
                     for p in load_curated(include_reserve=True,
                                           allow_placeholders=True)}
    except Exception:
        cat_by_id = {}
    out: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in store.query(
            "SELECT arm, prompt_id, winner FROM eval_results"
            " WHERE judge LIKE 'sonnet%' AND winner IN ('arm', 'base')"):
        category = ("harvested" if str(r["prompt_id"]).isdigit()
                    else cat_by_id.get(r["prompt_id"], "unknown"))
        cell = out[r["arm"]][category]
        cell[1] += 1
        cell[0] += r["winner"] == "arm"
    return {arm: {c: tuple(v) for c, v in cats.items()} for arm, cats in out.items()}


def build_paper_outputs(*, out_dir: Path, results_dir: Path | None = None,
                        store=None) -> list[Path]:
    from research.publish import pii_findings

    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    rows = protocol.load_rows(results_dir / "eval.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    md, records = pooled_table(rows)
    written = []
    (out_dir / "pooled_winrates.md").write_text(md)
    written.append(out_dir / "pooled_winrates.md")
    with open(out_dir / "pooled_winrates.csv", "w", newline="") as f:
        if records:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    written.append(out_dir / "pooled_winrates.csv")
    (out_dir / "agreement.md").write_text(agreement_table(rows, store))
    written.append(out_dir / "agreement.md")
    (out_dir / "timeline.md").write_text(timeline_table(rows))
    written.append(out_dir / "timeline.md")
    written += write_figures(rows, out_dir, store)

    # Figures are rendered from the same rows the tables carry, and SVG path
    # data is all digit runs (endless phone-like false positives) — so the
    # gate scans the text outputs, which contain every string a figure shows.
    findings = pii_findings([p for p in written if p.suffix in (".md", ".csv")])
    if findings:
        for p in written:
            p.unlink(missing_ok=True)
        raise SystemExit(f"PII gate blocked paper outputs: {findings[:5]} "
                         f"({len(findings)} finding(s)); nothing written")
    return written
