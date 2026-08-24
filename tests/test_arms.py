"""The single arm-construction path (research/arms.py).

Before this, nightly.stage_replay built real ArmSpecs while cli.cmd_eval had a
stub that printed "no artifact loader yet" and evaluated nothing, so
`python -m research eval --arms lora` silently did no work.
"""

from __future__ import annotations

import pytest

from research import artifacts
from research.arms import build_arm_specs
from research.db import ResearchStore


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


@pytest.fixture
def artifacts_dir(tmp_path):
    return tmp_path / "artifacts"


def _make_artifact(arm: str, artifacts_dir, date_str="20260814", files=None):
    version_dir = artifacts.new_version(arm, date_str, artifacts_dir)
    for name, content in (files or {}).items():
        (version_dir / name).write_text(content)
    artifacts.advance_current(arm, version_dir.name, artifacts_dir)
    return version_dir.name


def test_no_artifacts_yields_base_and_facts(store, artifacts_dir):
    """facts reads production's live store, so it needs no artifact."""
    specs, skipped = build_arm_specs(store, artifacts_dir=artifacts_dir)

    assert [s.name for s in specs] == ["base", "facts"]
    assert set(skipped) == {"prompt", "memory", "lora"}
    assert all(reason == "no current artifact" for reason in skipped.values())


def test_base_is_always_first(store, artifacts_dir):
    """Callers index specs[0] as the pairwise opponent."""
    specs, _ = build_arm_specs(store, artifacts_dir=artifacts_dir)
    assert specs[0].name == "base"
    assert specs[0].adapter_path is None
    assert specs[0].system_block == ""


def test_include_base_false_omits_it(store, artifacts_dir):
    specs, _ = build_arm_specs(store, artifacts_dir=artifacts_dir,
                               include_base=False)
    assert "base" not in [s.name for s in specs]


def test_prompt_arm_included_once_a_block_exists(store, artifacts_dir):
    version = _make_artifact("prompt", artifacts_dir,
                             files={"block.md": "- prefers oat milk"})

    specs, skipped = build_arm_specs(store, artifacts_dir=artifacts_dir)

    prompt = next(s for s in specs if s.name == "prompt")
    assert "prefers oat milk" in prompt.system_block
    assert prompt.system_block.startswith("LEARNED PREFERENCES:")
    assert prompt.artifact_version == version
    assert "prompt" not in skipped


def test_empty_prompt_block_does_not_create_an_arm(store, artifacts_dir):
    """An artifact dir with an empty block is not a usable arm."""
    _make_artifact("prompt", artifacts_dir, files={"block.md": "   \n"})

    specs, skipped = build_arm_specs(store, artifacts_dir=artifacts_dir)

    assert "prompt" not in [s.name for s in specs]
    assert skipped["prompt"] == "no current artifact"


def test_lora_arm_attaches_the_adapter_path(store, artifacts_dir):
    version = _make_artifact("lora", artifacts_dir,
                             files={"adapters.safetensors": "weights"})

    specs, _ = build_arm_specs(store, artifacts_dir=artifacts_dir)

    lora = next(s for s in specs if s.name == "lora")
    assert lora.artifact_version == version
    assert lora.adapter_path.endswith(version)
    assert (artifacts_dir / "lora" / version / "adapters.safetensors").exists()


def test_requesting_a_subset_returns_only_those_arms(store, artifacts_dir):
    _make_artifact("lora", artifacts_dir, files={"adapters.safetensors": "w"})
    _make_artifact("prompt", artifacts_dir, files={"block.md": "- x"})

    specs, skipped = build_arm_specs(store, artifacts_dir=artifacts_dir,
                                     arms=["lora"])

    assert [s.name for s in specs] == ["base", "lora"]
    assert skipped == {}


def test_unknown_arm_is_reported_not_silently_dropped(store, artifacts_dir):
    specs, skipped = build_arm_specs(store, artifacts_dir=artifacts_dir,
                                     arms=["lora", "nonsense"])

    assert skipped["nonsense"] == "unknown arm"
    assert "nonsense" not in [s.name for s in specs]


def test_requesting_base_explicitly_does_not_duplicate_it(store, artifacts_dir):
    specs, _ = build_arm_specs(store, artifacts_dir=artifacts_dir,
                               arms=["base", "facts"])
    assert [s.name for s in specs].count("base") == 1
