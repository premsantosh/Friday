"""The one place an arm's ArmSpec gets built from its artifacts.

There used to be two: the real logic in nightly.stage_replay, and a stub in
cli.cmd_eval that printed "arm 'lora' has no artifact loader yet" and evaluated
nothing. `python -m research eval --arms lora` silently did no work. Both now
call build_arm_specs, so the CLI and the nightly loop can never disagree about
what an arm is.

An arm participates when its `current` artifact pointer exists. Skipped arms are
returned with a reason rather than dropped, so callers report the truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from research import artifacts
from research.db import ResearchStore
from research.generate import ArmSpec

logger = logging.getLogger(__name__)

# base: vanilla model, the opponent every arm is judged against.
# facts: production's existing keyword-matched facts store, arm A's comparator,
#        so the study reports reflection-memory vs facts-store and not merely
#        vs vanilla. Neither is a study arm, and neither has an artifact.
BASELINE_ARMS = ("base", "facts")
STUDY_ARMS = ("prompt", "memory", "lora")
ALL_ARMS = BASELINE_ARMS + STUDY_ARMS


def build_arm_specs(
    store: ResearchStore,
    *,
    artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR,
    arms: Optional[list[str]] = None,
    include_base: bool = True,
    memory_index=None,
) -> tuple[list[ArmSpec], dict[str, str]]:
    """Build ArmSpecs for the requested arms.

    Args:
        arms: arm names to include; None means every arm that has an artifact,
            plus the `facts` comparator.
        include_base: prepend vanilla `base` (specs[0]), the pairwise opponent.
        memory_index: optional VectorIndex for arm A; built on demand when None.

    Returns:
        (specs, skipped) where skipped maps arm name to a human-readable reason.
    """
    requested = list(arms) if arms is not None else list(ALL_ARMS)
    requested = [a for a in requested if a != "base"]

    specs: list[ArmSpec] = [ArmSpec(name="base")] if include_base else []
    skipped: dict[str, str] = {}

    for arm in requested:
        if arm not in ALL_ARMS:
            skipped[arm] = "unknown arm"
            continue
        spec = _spec_for(arm, store, artifacts_dir, memory_index)
        if spec is None:
            skipped[arm] = "no current artifact"
            continue
        specs.append(spec)

    if skipped:
        logger.info("arms skipped: %s",
                    ", ".join(f"{k} ({v})" for k, v in sorted(skipped.items())))
    return specs, skipped


def _spec_for(arm: str, store: ResearchStore, artifacts_dir: Path,
              memory_index) -> Optional[ArmSpec]:
    """One arm's spec, or None when it has no artifact to contribute."""
    if arm == "facts":
        # No artifact: reads production's live facts store per query.
        from research.approaches import facts_baseline

        return ArmSpec(name="facts", block_for=facts_baseline.system_block_for)

    if arm == "prompt":
        from research.approaches import prompt_evolver

        block = prompt_evolver.system_block(artifacts_dir)
        if not block:
            return None
        return ArmSpec(
            name="prompt",
            system_block=block,
            artifact_version=artifacts.current_version("prompt", artifacts_dir),
        )

    if arm == "memory":
        version = artifacts.current_version("memory", artifacts_dir)
        if version is None:
            return None
        from research.approaches.memory_agent import ChromaIndex, MemoryAgent

        index = memory_index or ChromaIndex(
            Path(artifacts_dir).expanduser() / "memory_index")
        agent = MemoryAgent(store, artifacts_dir=artifacts_dir, index=index)
        # Pin retrieval to the snapshot's max id so the arm can't see memories
        # written after the version it is being evaluated as.
        max_id = agent.current_max_id()
        return ArmSpec(
            name="memory",
            block_for=lambda text: agent.system_block_for(text, max_id=max_id),
            artifact_version=version,
        )

    if arm == "lora":
        adapter = artifacts.current_path("lora", artifacts_dir)
        if adapter is None:
            return None
        return ArmSpec(
            name="lora",
            adapter_path=str(adapter),
            artifact_version=artifacts.current_version("lora", artifacts_dir),
        )

    return None
