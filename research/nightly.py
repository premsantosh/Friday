"""Nightly learning loop orchestrator.

Stages run in a fixed order, each isolated: a failing stage records its error
in runs.stage_status and later independent stages still run (arm B failing to
train must not stop arm C's eval). Heavy stages come after cheap ones.

    harvest -> reflect(A) -> evolve(C) -> train(B) -> replay -> eval -> report

Invoked by `python -m research nightly` (launchd at 03:30, or by hand).
--dry-run swaps in the FakeJudge, skips training and paid API calls.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from research import artifacts
from research.db import ResearchStore

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 36  # overlap across nights; feedback insertion is idempotent


@dataclass
class NightlyContext:
    store: ResearchStore
    date_str: str
    since_ts: float
    dry_run: bool = False
    artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR
    results: dict = field(default_factory=dict)  # cross-stage handoff


def stage_harvest(ctx: NightlyContext) -> str:
    from research.miners import mine_all

    exchanges = ctx.store.exchanges_since(ctx.since_ts)
    signals = mine_all(exchanges)
    inserted = 0
    for sig in signals:
        if ctx.store.has_feedback(sig.exchange_id, sig.source):
            continue
        ctx.store.add_feedback(sig.exchange_id, kind="implicit", signal=sig.signal,
                               source=sig.source, details=sig.details)
        inserted += 1

    backup_dir = Path(ctx.artifacts_dir).expanduser() / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"research-{ctx.date_str}.db"
    dest = sqlite3.connect(str(backup_path))
    with dest:
        ctx.store.conn.backup(dest)
    dest.close()
    backups = sorted(backup_dir.glob("research-*.db"))
    for old in backups[:-7]:
        old.unlink()

    return f"{len(exchanges)} exchanges, {inserted} new signals, backup {backup_path.name}"


def stage_reflect(ctx: NightlyContext) -> str:
    from research.approaches.memory_agent import ChromaIndex, MemoryAgent

    if ctx.dry_run:
        return "skipped (dry-run: no local LLM calls)"
    index = ChromaIndex(Path(ctx.artifacts_dir).expanduser() / "memory_index")
    agent = MemoryAgent(ctx.store, artifacts_dir=ctx.artifacts_dir, index=index)
    observed = agent.observe(ctx.since_ts)
    reflected = agent.maybe_reflect()
    version = agent.snapshot(ctx.date_str) if (observed or reflected) else None
    return (f"{observed} observations, {reflected} reflections"
            + (f", snapshot {version}" if version else ", no snapshot"))


def stage_evolve(ctx: NightlyContext) -> str:
    from research.approaches import prompt_evolver

    if ctx.dry_run:
        return "skipped (dry-run: no paid API calls)"
    version = prompt_evolver.evolve(ctx.store, ctx.since_ts, ctx.date_str,
                                    artifacts_dir=ctx.artifacts_dir)
    return f"advanced to {version}" if version else "no update (no data or unparseable)"


def stage_train(ctx: NightlyContext) -> str:
    from research.approaches.train_lora import train_nightly

    if ctx.dry_run:
        return "skipped (dry-run)"

    def sonnet(prompt: str) -> str:
        import anthropic
        resp = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5", max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return train_nightly(ctx.store, ctx.date_str, artifacts_dir=ctx.artifacts_dir,
                         correction_llm_fn=sonnet)


def stage_replay(ctx: NightlyContext) -> str:
    from research.approaches import prompt_evolver
    from research.generate import ArmSpec, generate_replay

    chat = [e for e in map(lambda r: ctx.store.get_exchange(r["id"]),
                           ctx.store.exchanges_since(ctx.since_ts))
            if e and e["route"] == "chat" and e.get("context_snapshot")]
    if not chat:
        return "no chat exchanges with snapshots"

    specs = [ArmSpec(name="base")]
    block = prompt_evolver.system_block(ctx.artifacts_dir)
    if block:
        specs.append(ArmSpec(
            name="prompt",
            system_block=block,
            artifact_version=artifacts.current_version("prompt", ctx.artifacts_dir),
        ))
    # 'facts' is arm A's baseline comparator: production's existing facts
    # store retrieved per query, so the study can report reflection-memory vs
    # facts-store rather than only vs vanilla.
    from research.approaches import facts_baseline

    specs.append(ArmSpec(name="facts", block_for=facts_baseline.system_block_for))

    memory_version = artifacts.current_version("memory", ctx.artifacts_dir)
    if memory_version is not None:
        from research.approaches.memory_agent import ChromaIndex, MemoryAgent

        index = ChromaIndex(Path(ctx.artifacts_dir).expanduser() / "memory_index")
        agent = MemoryAgent(ctx.store, artifacts_dir=ctx.artifacts_dir, index=index)
        max_id = agent.current_max_id()
        specs.append(ArmSpec(
            name="memory",
            block_for=lambda text: agent.system_block_for(text, max_id=max_id),
            artifact_version=memory_version,
        ))
    adapter = artifacts.current_path("lora", ctx.artifacts_dir)
    if adapter is not None:
        specs.append(ArmSpec(
            name="lora",
            adapter_path=str(adapter),
            artifact_version=artifacts.current_version("lora", ctx.artifacts_dir),
        ))

    generated = 0
    for spec in specs:
        responses = generate_replay(chat, spec)
        for eid, text in responses.items():
            ctx.store.add_shadow_response(
                eid, arm=spec.name, mode="replay", response_text=text,
                model_tag="mlx:Meta-Llama-3.1-8B-Instruct-4bit",
                artifact_version=spec.artifact_version,
            )
            generated += 1
    ctx.results["replay_exchange_ids"] = [e["id"] for e in chat]
    ctx.results["replay_arms"] = [s.name for s in specs]
    return f"{generated} candidates over {len(chat)} exchanges, arms {[s.name for s in specs]}"


def stage_eval(ctx: NightlyContext) -> str:
    from research.eval_runner import run_pairwise
    from research.evalset import EvalPrompt
    from research.judge import FakeJudge, SonnetJudge

    eids = ctx.results.get("replay_exchange_ids", [])
    arms = [a for a in ctx.results.get("replay_arms", []) if a != "base"]
    if not eids or not arms:
        return "nothing to judge"

    def responses_for(arm: str) -> dict[str, str]:
        rows = ctx.store.conn.execute(
            "SELECT exchange_id, response_text FROM shadow_responses"
            " WHERE arm = ? AND mode = 'replay' AND exchange_id IN"
            f" ({','.join('?' * len(eids))}) ORDER BY id",
            (arm, *eids),
        ).fetchall()
        return {str(r["exchange_id"]): r["response_text"] for r in rows}

    prompts = []
    for eid in eids:
        e = ctx.store.get_exchange(eid)
        prompts.append(EvalPrompt(id=str(eid), prompt=e["user_text"],
                                  category="preference_recall", source="harvested"))

    judge = FakeJudge() if ctx.dry_run else SonnetJudge()
    base_responses = responses_for("base")
    results = []
    for arm in arms:
        result = run_pairwise(prompts, responses_for(arm), base_responses, judge,
                              arm_name=arm, opponent_name="base")
        results.append(result)
        for o in result.outcomes:
            ctx.store.conn.execute(
                "INSERT INTO eval_results (run_id, arm, prompt_id, opponent, winner,"
                " judge, scores, ts) VALUES (?, ?, ?, 'base', ?, ?, ?, ?)",
                (ctx.results.get("run_id"), arm, o.prompt_id,
                 "arm" if o.score > 0.5 else ("base" if o.score < 0.5 else "tie"),
                 judge.name, json.dumps({"score": o.score}), time.time()),
            )
        ctx.store.conn.commit()
    ctx.results["pairwise"] = results
    return ", ".join(f"{r.arm}: {r.win_rate:.1%} (n={len(r.outcomes)})" for r in results)


def stage_report(ctx: NightlyContext) -> str:
    from research.report import append_csv_row, write_run_markdown

    results = ctx.results.get("pairwise", [])
    if not results:
        return "nothing to report"
    for r in results:
        append_csv_row(r, split="replay", date=ctx.date_str,
                       artifact_version=artifacts.current_version(r.arm, ctx.artifacts_dir) or "")
    path = write_run_markdown(results, split="replay", date=ctx.date_str)
    return str(path)


STAGES: list[tuple[str, Callable[[NightlyContext], str]]] = [
    ("harvest", stage_harvest),
    ("reflect", stage_reflect),
    ("evolve", stage_evolve),
    ("train", stage_train),
    ("replay", stage_replay),
    ("eval", stage_eval),
    ("report", stage_report),
]


def run_nightly(store: ResearchStore, *, dry_run: bool = False,
                date_str: Optional[str] = None,
                artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR,
                stages: Optional[list[str]] = None) -> dict[str, str]:
    """Run the loop; returns {stage: status}. Never raises out of a stage."""
    ctx = NightlyContext(
        store=store,
        date_str=date_str or datetime.now().strftime("%Y%m%d"),
        since_ts=time.time() - LOOKBACK_HOURS * 3600,
        dry_run=dry_run,
        artifacts_dir=artifacts_dir,
    )
    with store._lock:
        cur = store.conn.execute(
            "INSERT INTO runs (started_ts, stage_status) VALUES (?, '{}')",
            (time.time(),),
        )
        store.conn.commit()
    run_id = int(cur.lastrowid)
    ctx.results["run_id"] = run_id

    status: dict[str, str] = {}
    for name, fn in STAGES:
        if stages is not None and name not in stages:
            status[name] = "skipped (not selected)"
            continue
        t0 = time.monotonic()
        try:
            note = fn(ctx)
            status[name] = f"ok ({time.monotonic() - t0:.0f}s): {note}"
        except Exception as e:
            logger.exception("nightly stage %s failed", name)
            status[name] = f"FAILED: {type(e).__name__}: {e}"
        with store._lock:
            store.conn.execute(
                "UPDATE runs SET stage_status = ?, finished_ts = ? WHERE id = ?",
                (json.dumps(status), time.time(), run_id),
            )
            store.conn.commit()
    return status
