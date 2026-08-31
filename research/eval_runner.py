"""Pairwise eval tournament with position-swap and a paired sign test.

Each (prompt, armA, armB) pair is judged twice — once in each presentation
order — and the two verdicts average into one prompt score in [0, 1] for arm A.
"Improved" is pre-registered in research/data/evalset/PROTOCOL.md; this module
just computes the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Optional

from research.evalset import EvalPrompt
from research.judge import Judge, Verdict
from research.style import style_score


@dataclass
class PromptOutcome:
    prompt_id: str
    category: str
    score: float                 # arm A's position-swap-averaged score: 1 win, 0.5 tie, 0 loss
    verdicts: list[Verdict] = field(default_factory=list)


@dataclass
class PairwiseResult:
    arm: str                     # the challenger (side A)
    opponent: str                # the baseline (side B)
    judge: str
    outcomes: list[PromptOutcome]
    arm_style: float             # deterministic style compliance of the arm's responses
    opponent_style: float
    # Local-judge agreement audit (PROTOCOL.md): None = audit didn't run
    # (Ollama down / no shared decisive pairs) — never imputed as 0.0.
    local_agreement: Optional[float] = None
    n_audited: int = 0

    @property
    def n_decisive(self) -> int:
        return sum(1 for o in self.outcomes if o.score != 0.5)

    @property
    def wins(self) -> int:
        return sum(1 for o in self.outcomes if o.score > 0.5)

    @property
    def losses(self) -> int:
        return sum(1 for o in self.outcomes if o.score < 0.5)

    @property
    def win_rate(self) -> float:
        """Position-swap-averaged win rate over all judged prompts (ties = 0.5)."""
        if not self.outcomes:
            return 0.0
        return sum(o.score for o in self.outcomes) / len(self.outcomes)

    @property
    def p_value(self) -> float:
        return sign_test_p(self.wins, self.losses)

    def win_rate_for(self, category: str) -> float:
        subset = [o for o in self.outcomes if o.category == category]
        if not subset:
            return 0.5  # no data — neither pass nor fail; PROTOCOL treats 0.5 as neutral
        return sum(o.score for o in subset) / len(subset)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _score_pair(v_ab: Verdict, v_ba: Verdict) -> float:
    """Combine both presentation orders into arm A's score for one prompt.

    In the swapped call, arm A was shown as B — so a "B" verdict there is an
    arm-A win. Judge errors count as ties: unjudgeable must not look like wins.
    """
    def one(v: Verdict, a_label: str) -> float:
        if v.winner == a_label:
            return 1.0
        if v.winner == "tie" or v.winner == "error":
            return 0.5
        return 0.0

    return (one(v_ab, "A") + one(v_ba, "B")) / 2


def run_pairwise(
    prompts: list[EvalPrompt],
    arm_responses: dict[str, str],       # prompt_id -> response text
    opponent_responses: dict[str, str],
    judge: Judge,
    *,
    arm_name: str,
    opponent_name: str,
) -> PairwiseResult:
    outcomes = []
    for p in prompts:
        a = arm_responses.get(p.id)
        b = opponent_responses.get(p.id)
        if a is None or b is None:
            continue  # a generation failure drops the prompt, never fakes a verdict
        v_ab = judge.judge(p.prompt, p.annotations, a, b)
        v_ba = judge.judge(p.prompt, p.annotations, b, a)
        outcomes.append(PromptOutcome(
            prompt_id=p.id,
            category=p.category,
            score=_score_pair(v_ab, v_ba),
            verdicts=[v_ab, v_ba],
        ))
    judged_ids = {o.prompt_id for o in outcomes}
    return PairwiseResult(
        arm=arm_name,
        opponent=opponent_name,
        judge=judge.name,
        outcomes=outcomes,
        arm_style=style_score([arm_responses[i] for i in judged_ids]),
        opponent_style=style_score([opponent_responses[i] for i in judged_ids]),
    )


def agreement_rate(a: PairwiseResult, b: PairwiseResult) -> float:
    """Fraction of shared decisive prompts where two judges' directions agree."""
    return agreement_stats(a, b)[0] or 0.0


def agreement_stats(a: PairwiseResult, b: PairwiseResult) -> tuple[Optional[float], int]:
    """(direction-agreement rate, n shared decisive). Rate is None when there
    are no shared decisive pairs — an unmeasured audit must not read as 0%."""
    scores_b = {o.prompt_id: o.score for o in b.outcomes}
    shared = [
        (o.score, scores_b[o.prompt_id])
        for o in a.outcomes
        if o.prompt_id in scores_b and o.score != 0.5 and scores_b[o.prompt_id] != 0.5
    ]
    if not shared:
        return None, 0
    return sum(1 for x, y in shared if (x > 0.5) == (y > 0.5)) / len(shared), len(shared)
