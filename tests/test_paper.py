"""`research paper` output builders over synthetic rows (tables are pure;
figures need matplotlib and are skipped without it)."""

from __future__ import annotations

import pytest

from research.paper import agreement_table, build_paper_outputs, pooled_table, timeline_table


def _row(**over) -> dict:
    base = {
        "date": "2026-09-06", "run_id": "13", "arm": "memory", "opponent": "base",
        "artifact_version": "v20260905", "judge": "sonnet:claude-sonnet-5",
        "split": "curated", "n_prompts": "36", "n_decisive": "28",
        "wins": "18", "losses": "10", "win_rate": "0.610", "p_value": "0.090",
        "n_control": "10", "control_win_rate": "0.520",
        "arm_style": "0.900", "opponent_style": "0.900",
        "local_agreement": "0.750", "n_audited": "8", "notes": "",
    }
    base.update({k: str(v) for k, v in over.items()})
    return base


def _four_weeks(arm: str) -> list[dict]:
    return [_row(arm=arm, date=f"2026-09-{d:02d}") for d in (6, 13, 20, 27)]


def test_pooled_table_has_all_study_arms_and_holm():
    rows = _four_weeks("memory") + _four_weeks("prompt") + _four_weeks("lora")
    md, records = pooled_table(rows)
    assert "Holm" in md
    arms = {r["arm"] for r in records}
    assert arms == {"memory", "prompt", "lora"}
    memory = next(r for r in records if r["arm"] == "memory")
    assert memory["n"] == 4 * 36 and memory["wins"] == 4 * 18
    assert memory["holm_p"] != ""
    assert 0 < memory["wilson_low"] < memory["decisive_win_rate"] < memory["wilson_high"]


def test_agreement_table_reads_audit_columns():
    md = agreement_table([_row()], store=None)
    assert "75.0%" in md and "| 8 |" in md


def test_timeline_lists_versions_and_changelog():
    md = timeline_table([_row()])
    assert "v20260905" in md
    assert "2026-07-18" in md          # first PROTOCOL.md changelog entry


def test_build_outputs_and_pii_gate(tmp_path, monkeypatch):
    import csv

    results = tmp_path / "results"
    results.mkdir()
    rows = _four_weeks("memory")
    with open(results / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    out = tmp_path / "paper"
    written = build_paper_outputs(out_dir=out, results_dir=results)
    names = {p.name for p in written}
    assert {"pooled_winrates.md", "pooled_winrates.csv",
            "agreement.md", "timeline.md"} <= names

    # Poison the source: the gate must refuse and clean up.
    monkeypatch.setenv("FRIDAY_PII_NAMES", "memory")   # appears in every table
    with pytest.raises(SystemExit, match="PII gate"):
        build_paper_outputs(out_dir=tmp_path / "paper2", results_dir=results)
    assert not list((tmp_path / "paper2").glob("*.md"))
