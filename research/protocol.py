"""The pre-registered bar, in code.

research/data/evalset/PROTOCOL.md defines six conditions an arm must meet to
count as beating base. Until now that was prose a human applied by squinting at
results/eval.csv, which is exactly the sort of thing that drifts to fit the
result you were hoping for. This evaluates it mechanically.

Pure functions over CSV rows, so the whole thing is testable with synthetic
rows and never needs a model, a judge, or a database.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from research.report import RESULTS_DIR

# Thresholds are the pre-registered ones. Changing a number here is a protocol
# change and belongs in PROTOCOL.md's changelog.
MIN_WIN_RATE = 0.55
MAX_P_VALUE = 0.05
MIN_N = 30
MIN_CONTROL_WIN_RATE = 0.45
MAX_STYLE_REGRESSION = 0.05
CONSECUTIVE_EVALS = 2
EXCLUDED_JUDGES = frozenset({"fake"})  # dry-run rows, not evidence
MIN_DAYS_BETWEEN_EVALS = 5  # "weekly", with slack for a run that slipped a day


@dataclass
class Condition:
    number: int
    name: str
    passed: Optional[bool]  # None = not yet measurable
    detail: str


@dataclass
class BarResult:
    arm: str
    conditions: list[Condition]

    @property
    def improved(self) -> bool:
        """True only when every condition is affirmatively met."""
        return all(c.passed is True for c in self.conditions)

    def summary(self) -> str:
        if self.improved:
            return "BEATS BASE (all 6 conditions met)"
        failed = [c.number for c in self.conditions if c.passed is False]
        pending = [c.number for c in self.conditions if c.passed is None]
        parts = []
        if failed:
            parts.append(f"fails {failed}")
        if pending:
            parts.append(f"pending {pending}")
        return "; ".join(parts) or "no verdict"


def load_rows(csv_path: Path | None = None) -> list[dict]:
    """Every row of results/eval.csv, newest last."""
    # Resolved at call time so tests and --results-dir can redirect it.
    path = Path(csv_path) if csv_path is not None else RESULTS_DIR / "eval.csv"
    if not path.exists():
        raise FileNotFoundError(f"no results csv at {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _i(row: dict, key: str) -> int:
    value = _f(row, key)
    return int(value) if value is not None else 0


def _date(row: dict) -> Optional[datetime]:
    raw = (row.get("date") or "").strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _conditions_1_to_5(row: dict) -> list[Condition]:
    """Conditions 1-5 for a single eval row."""
    win_rate = _f(row, "win_rate")
    p_value = _f(row, "p_value")
    n = _i(row, "n_prompts")
    n_control = _i(row, "n_control")
    control = _f(row, "control_win_rate")
    arm_style = _f(row, "arm_style")
    opp_style = _f(row, "opponent_style")

    out = [
        Condition(1, f"win rate > {MIN_WIN_RATE:.0%}",
                  None if win_rate is None else win_rate > MIN_WIN_RATE,
                  "no data" if win_rate is None else f"{win_rate:.1%}"),
        Condition(2, f"sign test p < {MAX_P_VALUE}",
                  None if p_value is None else p_value < MAX_P_VALUE,
                  "no data" if p_value is None else f"p={p_value:.3f}"),
        Condition(3, f"n >= {MIN_N}", n >= MIN_N, f"n={n}"),
    ]

    # An unmeasured control is pending, never a pass. The neutral 0.500 that a
    # split with no control probes would otherwise produce clears the >= 45%
    # bar for free, which would make this condition decorative.
    if not n_control or control is None:
        out.append(Condition(4, f"generic_control >= {MIN_CONTROL_WIN_RATE:.0%}",
                             None, "no control probes in this split"))
    else:
        out.append(Condition(4, f"generic_control >= {MIN_CONTROL_WIN_RATE:.0%}",
                             control >= MIN_CONTROL_WIN_RATE,
                             f"{control:.1%} over {n_control} probe(s)"))

    if arm_style is None or opp_style is None:
        out.append(Condition(5, "no style regression", None, "no data"))
    else:
        delta = arm_style - opp_style
        out.append(Condition(
            5, "no style regression", delta >= -MAX_STYLE_REGRESSION,
            f"arm {arm_style:.2f} vs base {opp_style:.2f} (delta {delta:+.2f})"))
    return out


def evaluate_bar(rows: list[dict], *, arm: str,
                 split: str = "curated") -> BarResult:
    """Where `arm` stands against the six pre-registered conditions.

    Conditions 1-5 are read off the most recent (arm, split) row. Condition 6
    additionally requires the previous eval at least MIN_DAYS_BETWEEN_EVALS
    earlier to have met 1-5 as well.
    """
    # `--dry-run` nightlies write FakeJudge rows so the end-to-end harness is
    # exercised; they are not evidence and must never count toward the bar.
    matching = [r for r in rows
                if r.get("arm") == arm and r.get("split") == split
                and r.get("judge") not in EXCLUDED_JUDGES]
    matching.sort(key=lambda r: (_date(r) or datetime.min))

    if not matching:
        pending = [Condition(n, name, None, "no eval rows yet") for n, name in (
            (1, f"win rate > {MIN_WIN_RATE:.0%}"),
            (2, f"sign test p < {MAX_P_VALUE}"),
            (3, f"n >= {MIN_N}"),
            (4, f"generic_control >= {MIN_CONTROL_WIN_RATE:.0%}"),
            (5, "no style regression"),
            (6, f"holds on {CONSECUTIVE_EVALS} consecutive weekly evals"),
        )]
        return BarResult(arm=arm, conditions=pending)

    latest = matching[-1]
    conditions = _conditions_1_to_5(latest)

    latest_ok = all(c.passed is True for c in conditions)
    latest_date = _date(latest)
    prior_ok = None
    prior_detail = "no earlier weekly eval"
    for row in reversed(matching[:-1]):
        row_date = _date(row)
        if latest_date and row_date and \
                (latest_date - row_date).days < MIN_DAYS_BETWEEN_EVALS:
            continue  # same week; condition 6 is about *consecutive weeks*
        prior_ok = all(c.passed is True for c in _conditions_1_to_5(row))
        prior_detail = f"previous eval {row.get('date')}: " \
                       f"{'met 1-5' if prior_ok else 'did not meet 1-5'}"
        break

    if prior_ok is None:
        # Nothing to compare against yet: pending, not failed.
        condition_6 = Condition(6, f"holds on {CONSECUTIVE_EVALS} consecutive weekly evals",
                                None, f"1 of {CONSECUTIVE_EVALS} evals ({prior_detail})")
    else:
        condition_6 = Condition(6, f"holds on {CONSECUTIVE_EVALS} consecutive weekly evals",
                                bool(latest_ok and prior_ok), prior_detail)
    conditions.append(condition_6)
    return BarResult(arm=arm, conditions=conditions)
