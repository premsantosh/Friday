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


# ------------------------------------------------------------ judge filter
def test_fake_judge_rows_never_count_toward_the_bar():
    """A --dry-run nightly appends judge=fake rows to results/eval.csv so the
    harness runs end to end; they must read as 'no eval rows yet', not as a
    pass (or a fail) of the pre-registered bar."""
    rows = [_row(date="2026-08-07", judge="fake"), _row(date="2026-08-14", judge="fake")]
    bar = evaluate_bar(rows, arm="lora")
    assert not bar.improved
    assert all(c.passed is None for c in bar.conditions)

    # A real judge row still counts, and the fake ones do not sneak in as the
    # 'previous eval' for condition 6.
    rows.append(_row(date="2026-08-21"))
    bar = evaluate_bar(rows, arm="lora")
    assert [_cond(bar, n).passed for n in (1, 2, 3, 4, 5)] == [True] * 5
    assert _cond(bar, 6).passed is not True


# ----------------------------------------------- pooled primary endpoint (W7)
def test_wilson_interval_known_values():
    from research.protocol import wilson_interval

    lo, hi = wilson_interval(70, 128)
    assert (round(lo, 4), round(hi, 4)) == (0.4605, 0.6305)
    lo, hi = wilson_interval(18, 32)
    assert (round(lo, 4), round(hi, 4)) == (0.3933, 0.7183)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_holm_hand_worked():
    from research.protocol import holm

    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # step-down: a: 3*0.01=0.03; c: max(0.03, 2*0.03)=0.06; b: max(0.06, 1*0.04)=0.06
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert holm({}) == {}


def test_evaluate_pooled_sums_weekly_curated_rows():
    from research.protocol import evaluate_pooled

    rows = [
        _row(date="2026-08-03", n_prompts="32", n_decisive="24", wins="14", losses="10",
             win_rate="0.560", n_control="10", control_win_rate="0.500"),
        _row(date="2026-08-10", n_prompts="32", n_decisive="26", wins="16", losses="10",
             win_rate="0.590", n_control="10", control_win_rate="0.520"),
        # Same-week duplicate of the 08-10 eval: must be skipped.
        _row(date="2026-08-11", n_prompts="32", n_decisive="30", wins="30", losses="0",
             win_rate="0.990"),
        _row(date="2026-08-17", n_prompts="36", n_decisive="28", wins="20", losses="8",
             win_rate="0.610", n_control="10", control_win_rate="0.480"),
        # FakeJudge row: never evidence.
        _row(date="2026-08-18", judge="fake", wins="99", losses="0"),
    ]
    pooled = evaluate_pooled(rows, arm="lora", window=4)
    # picked: 08-03, 08-11 (later of the same-week pair), 08-17
    assert len(pooled.dates) == 3
    assert pooled.n_prompts == 32 + 32 + 36
    assert pooled.wins == 14 + 30 + 20
    assert pooled.losses == 10 + 0 + 8
    # every picked row carries the helper's default 10 control probes
    assert pooled.n_control == 30
    assert round(pooled.control_win_rate, 3) == round((0.5 * 20 + 0.48 * 10) / 30, 3)
    assert 0 < pooled.wilson_low < pooled.decisive_win_rate < pooled.wilson_high <= 1
    assert "pooled" in pooled.summary()


def test_evaluate_pooled_replay_excludes_pre_prospective_rows():
    from research.protocol import REPLAY_PROSPECTIVE_FROM, evaluate_pooled

    assert REPLAY_PROSPECTIVE_FROM == "2026-08-31"
    rows = [
        _row(date="2026-08-25", split="replay", wins="4", losses="0"),   # contaminated
        _row(date="2026-09-02", split="replay", n_prompts="4", n_decisive="3",
             wins="2", losses="1", win_rate="0.620"),
        _row(date="2026-09-03", split="replay", n_prompts="6", n_decisive="4",
             wins="3", losses="1", win_rate="0.580"),
    ]
    pooled = evaluate_pooled(rows, arm="lora", split="replay")
    assert pooled.dates == ["2026-09-02", "2026-09-03"]
    assert pooled.wins == 5 and pooled.losses == 2


def test_evaluate_pooled_no_data():
    from research.protocol import evaluate_pooled

    pooled = evaluate_pooled([], arm="lora")
    assert pooled.summary() == "no pooled data yet"
    assert pooled.n_prompts == 0
