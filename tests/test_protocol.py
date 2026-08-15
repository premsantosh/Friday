"""The pre-registered bar (research/protocol.py), over synthetic CSV rows."""

from __future__ import annotations

from research.protocol import BarResult, evaluate_bar, load_rows


def _row(**over) -> dict:
    """A row that passes conditions 1-5 unless overridden."""
    base = {
        "date": "2026-08-14", "run_id": "1", "arm": "lora", "opponent": "base",
        "artifact_version": "v20260814", "judge": "claude-sonnet-5",
        "split": "curated", "n_prompts": "34", "n_decisive": "30",
        "wins": "20", "losses": "10", "win_rate": "0.600", "p_value": "0.030",
        "n_control": "10", "control_win_rate": "0.500",
        "arm_style": "0.850", "opponent_style": "0.840",
    }
    base.update({k: str(v) for k, v in over.items()})
    return base


def _cond(bar: BarResult, number: int):
    return next(c for c in bar.conditions if c.number == number)


# ------------------------------------------------------------ conditions 1-5
def test_all_conditions_pass_on_two_weekly_evals():
    rows = [_row(date="2026-08-07"), _row(date="2026-08-14")]
    bar = evaluate_bar(rows, arm="lora")

    assert bar.improved
    assert bar.summary() == "BEATS BASE (all 6 conditions met)"
    assert all(c.passed is True for c in bar.conditions)


def test_win_rate_at_the_threshold_does_not_pass():
    """The bar is > 55%, not >= 55%."""
    rows = [_row(win_rate="0.550")]
    assert _cond(evaluate_bar(rows, arm="lora"), 1).passed is False


def test_p_value_too_high_fails_condition_2():
    rows = [_row(p_value="0.080")]
    bar = evaluate_bar(rows, arm="lora")
    assert _cond(bar, 2).passed is False
    assert not bar.improved


def test_too_few_prompts_fails_condition_3():
    rows = [_row(n_prompts="12")]
    bar = evaluate_bar(rows, arm="lora")
    assert _cond(bar, 3).passed is False
    assert "n=12" in _cond(bar, 3).detail


def test_control_regression_fails_condition_4():
    rows = [_row(control_win_rate="0.300")]
    assert _cond(evaluate_bar(rows, arm="lora"), 4).passed is False


def test_missing_control_probes_is_pending_not_a_pass():
    """A split with no control probes must not clear the bar for free.

    The replay split has no generic_control category, and win_rate_for returns
    a neutral 0.5 for "no data" — which would sail past the >= 45% threshold
    and make the no-regression condition decorative.
    """
    rows = [_row(split="replay", n_control="0", control_win_rate="")]
    bar = evaluate_bar(rows, arm="lora", split="replay")

    assert _cond(bar, 4).passed is None
    assert "no control probes" in _cond(bar, 4).detail
    assert not bar.improved


def test_style_regression_beyond_tolerance_fails_condition_5():
    rows = [_row(arm_style="0.700", opponent_style="0.850")]
    bar = evaluate_bar(rows, arm="lora")
    assert _cond(bar, 5).passed is False

    # Within the 0.05 tolerance is still a pass.
    rows = [_row(arm_style="0.810", opponent_style="0.850")]
    assert _cond(evaluate_bar(rows, arm="lora"), 5).passed is True


# -------------------------------------------------------------- condition 6
def test_single_eval_leaves_condition_6_pending():
    bar = evaluate_bar([_row()], arm="lora")

    assert all(_cond(bar, n).passed is True for n in (1, 2, 3, 4, 5))
    assert _cond(bar, 6).passed is None
    assert "1 of 2" in _cond(bar, 6).detail
    assert not bar.improved, "pending is not passing"


def test_two_evals_in_the_same_week_do_not_satisfy_condition_6():
    """Consecutive *weekly* evals; two runs a day apart are one week."""
    rows = [_row(date="2026-08-13"), _row(date="2026-08-14")]
    assert _cond(evaluate_bar(rows, arm="lora"), 6).passed is None


def test_condition_6_fails_when_the_previous_eval_missed():
    rows = [_row(date="2026-08-07", win_rate="0.400"), _row(date="2026-08-14")]
    bar = evaluate_bar(rows, arm="lora")

    assert _cond(bar, 6).passed is False
    assert "did not meet 1-5" in _cond(bar, 6).detail
    assert not bar.improved


def test_condition_6_fails_when_the_latest_eval_missed():
    rows = [_row(date="2026-08-07"), _row(date="2026-08-14", p_value="0.400")]
    bar = evaluate_bar(rows, arm="lora")
    assert _cond(bar, 6).passed is False


# ------------------------------------------------------------------ plumbing
def test_rows_for_other_arms_and_splits_are_ignored():
    rows = [
        _row(date="2026-08-07", arm="prompt"),
        _row(date="2026-08-07", split="replay"),
        _row(date="2026-08-14"),
    ]
    bar = evaluate_bar(rows, arm="lora", split="curated")
    assert _cond(bar, 6).passed is None, "only one curated lora row exists"


def test_no_rows_yields_all_pending():
    bar = evaluate_bar([], arm="lora")
    assert len(bar.conditions) == 6
    assert all(c.passed is None for c in bar.conditions)
    assert not bar.improved
    assert "pending" in bar.summary()


def test_summary_lists_failures_and_pendings():
    rows = [_row(win_rate="0.400", n_control="0", control_win_rate="")]
    summary = evaluate_bar(rows, arm="lora").summary()
    assert "fails [1]" in summary
    assert "pending [4, 6]" in summary


def test_load_rows_reads_the_csv(tmp_path):
    path = tmp_path / "eval.csv"
    path.write_text("date,arm,split,win_rate\n2026-08-14,lora,curated,0.6\n")
    rows = load_rows(path)
    assert rows[0]["arm"] == "lora"
    assert rows[0]["win_rate"] == "0.6"
