"""Batch candidate generation for eval/replay via mlx_lm.

All arms generate through the same pinned 4-bit base with greedy decoding, so
the only delta between arms is their artifact: an adapter (B), a retrieved
memory block (A), or a learned-preferences block (C). Ollama is never used
here — one runtime removes sampler/quantization confounds.

mlx_lm is imported lazily so the test suite never needs model weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from research.evalset import EvalPrompt
from research.persona import PERSONA_PROMPT

logger = logging.getLogger(__name__)

BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
MAX_TOKENS = 200  # persona targets <=3 sentences; hard cap for runaway generations


@dataclass
class ArmSpec:
    """What makes an arm's generation different from vanilla base."""
    name: str                                             # base | memory | lora | prompt
    adapter_path: Optional[str] = None                    # arm B
    system_block: str = ""                                # arm C: learned-preferences block
    block_for: Optional[Callable[[str], str]] = None      # arm A: per-query retrieval
    artifact_version: Optional[str] = None

    def _blocks(self, user_text: str) -> list[str]:
        parts = []
        if self.system_block:
            parts.append(self.system_block)
        if self.block_for is not None:
            block = self.block_for(user_text)
            if block:
                parts.append(block)
        return parts

    def system_prompt_for(self, user_text: str) -> str:
        return "\n\n".join([PERSONA_PROMPT, *self._blocks(user_text)])


def generate_replay(
    exchanges: list[dict],
    arm: ArmSpec,
    *,
    model: str = BASE_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> dict[int, str]:
    """exchange_id -> response, regenerating from each exchange's context
    snapshot (the exact system prompt + history production saw). All arms see
    the identical snapshot; the arm's artifact is appended/loaded on top.
    """
    from mlx_lm import generate, load

    logger.info("Replay: loading %s (adapter=%s) for arm %s", model, arm.adapter_path,
                arm.name)
    lm, tokenizer = load(model, adapter_path=arm.adapter_path)

    out: dict[int, str] = {}
    for e in exchanges:
        snapshot = e.get("context_snapshot")
        if not snapshot:
            continue
        try:
            system = "\n\n".join([snapshot.get("system_prompt", PERSONA_PROMPT),
                                  *arm._blocks(e.get("user_text", ""))])
            chat = [{"role": "system", "content": system}, *snapshot.get("messages", [])]
            templated = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            out[e["id"]] = generate(lm, tokenizer, prompt=templated,
                                    max_tokens=max_tokens).strip()
        except Exception:
            logger.warning("Replay generation failed for exchange %s (arm %s)",
                           e.get("id"), arm.name, exc_info=True)
    return out


def generate_candidates(
    prompts: list[EvalPrompt],
    arm: ArmSpec,
    *,
    model: str = BASE_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, str]:
    """prompt_id -> response for one arm. Loads the model once per call.

    Greedy decoding (mlx_lm default sampler, temperature 0) — reproducible and
    fair across arms. A per-prompt failure drops that prompt with a log line
    rather than aborting the batch.
    """
    from mlx_lm import generate, load

    logger.info("Loading %s (adapter=%s) for arm %s", model, arm.adapter_path, arm.name)
    lm, tokenizer = load(model, adapter_path=arm.adapter_path)

    out: dict[str, str] = {}
    for p in prompts:
        try:
            chat = [
                {"role": "system", "content": arm.system_prompt_for(p.prompt)},
                {"role": "user", "content": p.prompt},
            ]
            templated = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            out[p.id] = generate(lm, tokenizer, prompt=templated, max_tokens=max_tokens).strip()
        except Exception:
            logger.warning("Generation failed for prompt %s (arm %s)", p.id, arm.name,
                           exc_info=True)
    return out
