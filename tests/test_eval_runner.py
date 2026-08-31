"""Tests for the eval harness: runner math, judges parsing, evalset, reports.

Hermetic: FakeJudge only, tmp_path for reports, no model loads.
"""

from __future__ import annotations

import csv

from research.eval_runner import agreement_rate, run_pairwise, sign_test_p
from research.evalset import EvalPrompt, load_curated
from research.judge import FakeJudge, Verdict, _parse_verdict
from research.report import append_csv_row, write_run_markdown
from research.style import check_style


def _prompts(n, category="style"):
    return [EvalPrompt(id=f"p{i}", prompt=f"question {i}", category=category)
            for i in range(n)]


# ------------------------------------------------------------------ sign test

def test_sign_test_no_data_is_1():
    assert sign_test_p(0, 0) == 1.0


def test_sign_test_balanced_is_1():
    assert sign_test_p(5, 5) > 0.99


def test_sign_test_lopsided_is_significant():
    assert sign_test_p(29, 1) < 0.001


# -------------------------------------------------------------- position swap

def test_position_swap_averages_both_orders():
    # Judge scripted: first call (A=arm) says A, swapped call says A (i.e. the
    # opponent) — net 0.5 for the prompt: position bias cancels to a tie.
    judge = FakeJudge(script=["A", "A"])
    prompts = _prompts(1)
    result = run_pairwise(prompts, {"p0": "arm"}, {"p0": "opp"}, judge,
                          arm_name="lora", opponent_name="base")
    assert result.outcomes[0].score == 0.5
    assert result.n_decisive == 0


def test_consistent_win_across_orders_scores_1():
    # Arm wins as A in order 1 and as B in order 2.
    judge = FakeJudge(script=["A", "B"])
    result = run_pairwise(_prompts(1), {"p0": "arm"}, {"p0": "opp"}, judge,
                          arm_name="lora", opponent_name="base")
    assert result.outcomes[0].score == 1.0
    assert result.wins == 1


def test_judge_error_counts_as_tie():
    judge = FakeJudge(script=["error", "error"])
    result = run_pairwise(_prompts(1), {"p0": "arm"}, {"p0": "opp"}, judge,
                          arm_name="lora", opponent_name="base")
    assert result.outcomes[0].score == 0.5


def test_missing_candidate_drops_prompt():
    judge = FakeJudge()
    result = run_pairwise(_prompts(2), {"p0": "arm only"}, {"p0": "opp", "p1": "opp"},
                          judge, arm_name="lora", opponent_name="base")
    assert [o.prompt_id for o in result.outcomes] == ["p0"]


def test_category_win_rate_and_control():
    prompts = _prompts(2) + [EvalPrompt(id="c0", prompt="ctl", category="generic_control")]
    judge = FakeJudge(script=["A", "B", "A", "B", "A", "B"])  # arm wins all three
    result = run_pairwise(
        prompts,
        {"p0": "a", "p1": "a", "c0": "a"},
        {"p0": "b", "p1": "b", "c0": "b"},
        judge, arm_name="lora", opponent_name="base",
    )
    assert result.win_rate == 1.0
    assert result.win_rate_for("generic_control") == 1.0
    assert result.win_rate_for("routine") == 0.5  # no data -> neutral


def test_agreement_rate():
    j1 = FakeJudge(script=["A", "B", "A", "B"])  # arm wins p0 and p1
    j2 = FakeJudge(script=["A", "B", "B", "A"])  # arm wins p0, loses p1
    r1 = run_pairwise(_prompts(2), {"p0": "a", "p1": "a"}, {"p0": "b", "p1": "b"},
                      j1, arm_name="x", opponent_name="base")
    r2 = run_pairwise(_prompts(2), {"p0": "a", "p1": "a"}, {"p0": "b", "p1": "b"},
                      j2, arm_name="x", opponent_name="base")
    assert agreement_rate(r1, r2) == 0.5


# ------------------------------------------------------------- verdict parsing

def test_parse_verdict_tolerates_prose():
    v = _parse_verdict('Sure. {"winner": "B", "reason": "honors preference"} Done.', "j")
    assert (v.winner, v.reason) == ("B", "honors preference")


def test_parse_verdict_rejects_garbage():
    assert _parse_verdict("no json here", "j").winner == "error"
    assert _parse_verdict('{"winner": "C"}', "j").winner == "error"


# ------------------------------------------------------------------ style

def test_style_checks():
    good = check_style("Very good, sir. The oven is preheating.")
    assert good.compliant and good.addresses_user
    bad = check_style("Wow!! Amazing! So exciting! Let me tell you all about it! " * 3)
    assert not bad.compliant


# ------------------------------------------------------------------ evalset

def test_curated_evalset_loads_and_validates():
    # allow_placeholders: the personalization probes are still FILL-IN, which
    # load_curated refuses by default (see test_evalset.py).
    prompts = load_curated(allow_placeholders=True)
    assert len(prompts) >= 15
    ids = [p.id for p in prompts]
    assert len(ids) == len(set(ids))
    assert any(p.category == "generic_control" for p in prompts)


# ------------------------------------------------------------------- reports

def test_csv_and_markdown_report(tmp_path):
    judge = FakeJudge(script=["A", "B"])
    result = run_pairwise(_prompts(1), {"p0": "arm"}, {"p0": "opp"}, judge,
                          arm_name="lora", opponent_name="base")
    csv_path = append_csv_row(result, split="curated", date="2026-07-18",
                              artifact_version="v20260718", results_dir=tmp_path)
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["arm"] == "lora"
    assert rows[0]["win_rate"] == "1.000"
    assert rows[0]["artifact_version"] == "v20260718"

    # Appending again grows the file without re-writing the header.
    append_csv_row(result, split="curated", date="2026-07-19", results_dir=tmp_path)
    assert len(list(csv.DictReader(open(csv_path)))) == 2

    md_path = write_run_markdown([result], split="curated", date="2026-07-18",
                                 results_dir=tmp_path)
    text = md_path.read_text()
    assert "| lora | base |" in text
    assert "100.0%" in text


# ------------------------------------------------------------ response parsing
class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, *blocks):
        self.content = list(blocks)


def test_first_text_skips_thinking_blocks():
    """Sonnet 5 thinks adaptively by default: content[0] can be a ThinkingBlock
    (the 2026-08-23 nightly's evolve stage died on .content[0].text)."""
    from research.judge import first_text

    resp = _Resp(_Block("thinking", thinking="…"), _Block("text", text="the answer"))
    assert first_text(resp) == "the answer"
    assert first_text(_Resp(_Block("thinking", thinking="…"))) == ""
    assert first_text(_Resp()) == ""


def test_agreement_stats_counts_shared_decisive_only():
    from research.eval_runner import PairwiseResult, PromptOutcome, agreement_stats

    def result(scores):
        return PairwiseResult(arm="a", opponent="base", judge="j",
                              outcomes=[PromptOutcome(str(i), "style", s)
                                        for i, s in enumerate(scores)],
                              arm_style=1.0, opponent_style=1.0)

    a = result([1.0, 0.0, 0.5, 1.0])
    b = result([1.0, 1.0, 1.0, 0.5])
    # shared decisive: prompts 0 (agree) and 1 (disagree); 2 is a tie for a,
    # 3 is a tie for b.
    rate, n = agreement_stats(a, b)
    assert (rate, n) == (0.5, 2)

    all_ties = result([0.5, 0.5, 0.5, 0.5])
    rate, n = agreement_stats(a, all_ties)
    assert rate is None and n == 0        # unmeasured, not 0%
