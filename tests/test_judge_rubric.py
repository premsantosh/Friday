"""The judge rubric is a measurement instrument: its text is pinned by hash.

Editing _RUBRIC without bumping RUBRIC_VERSION fails here. Bumping the version
requires adding the new sha below — deliberate friction, and the paper reports
results per rubric version.
"""

import hashlib

from research.judge import _RUBRIC, JUDGE_TEMPERATURE, RUBRIC_VERSION, rubric_id

PINNED = {
    "r1": "a23245c5625230b8f031ca9bba05a2e71148bed3bf269b23191872be22103002",
}


def test_rubric_text_matches_pinned_sha():
    assert RUBRIC_VERSION in PINNED, (
        f"RUBRIC_VERSION {RUBRIC_VERSION!r} has no pinned sha — add it here "
        "in the same change that bumps the version.")
    actual = hashlib.sha256(_RUBRIC.encode()).hexdigest()
    assert actual == PINNED[RUBRIC_VERSION], (
        "_RUBRIC text changed without bumping RUBRIC_VERSION. The rubric is a "
        "measurement instrument: bump the version and pin the new sha.")


def test_rubric_id_format():
    assert rubric_id() == f"{RUBRIC_VERSION}:{PINNED[RUBRIC_VERSION][:8]}"


def test_judge_temperature_is_pinned_to_zero():
    assert JUDGE_TEMPERATURE == 0.0
