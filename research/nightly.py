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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from research import artifacts
from research.db import ResearchStore

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 36  # overlap across nights; feedback insertion is idempotent
# Generation is the expensive stage and the corpus only grows. At the measured
# ~14 tok/s (research/scripts/PINNED.md), 25 exchanges x 5 arms x 200 tokens is
# roughly half an hour, which fits comfortably in the 03:30 window.
REPLAY_MAX_EXCHANGES = 25


def _dry_replay(exchanges: list[dict], arm) -> dict[int, str]:
    """Deterministic stand-in for mlx generation, so --dry-run exercises the
    whole pipeline (replay -> eval -> report) without loading model weights."""
    return {e["id"]: f"[{arm.name}] {e.get('user_text', '')[:80]}" for e in exchanges}


def _dry_candidates(prompts: list, arm) -> dict[str, str]:
    return {p.id: f"[{arm.name}] {p.prompt[:80]}" for p in prompts}


@dataclass
class NightlyContext:
    store: ResearchStore
    date_str: str
    since_ts: float
    dry_run: bool = False
    weekly: bool = False       # also judge the curated split
    artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR
    results_dir: Optional[Path] = None  # None = the repo's results/
    results: dict = field(default_factory=dict)  # cross-stage handoff
    # Injected so tests and --dry-run can drive the real stage logic.
    replay_fn: Callable = None
    candidates_fn: Callable = None

    def __post_init__(self) -> None:
        if self.replay_fn is None or self.candidates_fn is None:
            if self.dry_run:
                replay, candidates = _dry_replay, _dry_candidates
            else:
                from research.generate import generate_candidates, generate_replay

                replay, candidates = generate_replay, generate_candidates
            self.replay_fn = self.replay_fn or replay
            self.candidates_fn = self.candidates_fn or candidates


def stage_harvest(ctx: NightlyContext) -> str:
    from research.miners import mine_all

    exchanges = ctx.store.exchanges_since(ctx.since_ts)
    signals = mine_all(exchanges)
    inserted = 0
    for sig in signals:
        if ctx.store.has_feedback(sig.exchange_id, sig.source):
            continue
        # add_feedback emits feedback.added; this records that a miner is what
        # produced the signal, and on what evidence.
        ctx.store.emit("feedback.mined", subject_type="exchange",
                       subject_id=sig.exchange_id,
                       detail={"source": sig.source, "signal": sig.signal,
                               "details": sig.details})
        ctx.store.add_feedback(sig.exchange_id, kind="implicit", signal=sig.signal,
                               source=sig.source, details=sig.details)
        inserted += 1

    backup_dir = Path(ctx.artifacts_dir).expanduser() / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"research-{ctx.date_str}.db"
    ctx.store.backup_to(backup_path)
    backups = sorted(backup_dir.glob("research-*.db"))
    pruned = [p.name for p in backups[:-7]]
    for old in backups[:-7]:
        old.unlink()
    ctx.store.emit("db.backed_up", subject_type="run",
                   subject_id=ctx.results.get("run_id"),
                   detail={"path": backup_path.name,
                           "bytes": backup_path.stat().st_size,
                           "pruned": pruned})

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
        from research.judge import first_text
        resp = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5", max_tokens=400,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        return first_text(resp)

    return train_nightly(ctx.store, ctx.date_str, artifacts_dir=ctx.artifacts_dir,
                         correction_llm_fn=sonnet)


def stage_replay(ctx: NightlyContext) -> str:
    from research.arms import build_arm_specs
    from research.generate import BASE_MODEL_TAG

    chat = [e for e in map(lambda r: ctx.store.get_exchange(r["id"]),
                           ctx.store.exchanges_since(ctx.since_ts))
            if e and e["route"] == "chat" and e.get("context_snapshot")]
    if not chat:
        return "no chat exchanges with snapshots"
    # Bounded work per night: generation is the expensive stage and the corpus
    # only grows. Most recent first so the window stays current.
    dropped = 0
    if len(chat) > REPLAY_MAX_EXCHANGES:
        dropped = len(chat) - REPLAY_MAX_EXCHANGES
        chat = chat[-REPLAY_MAX_EXCHANGES:]

    specs, skipped = build_arm_specs(ctx.store, artifacts_dir=ctx.artifacts_dir)

    generated = 0
    for spec in specs:
        responses = ctx.replay_fn(chat, spec)
        for e in chat:
            text = responses.get(e["id"])
            if not text:
                ctx.store.emit("replay.failed", subject_type="exchange",
                               subject_id=e["id"], arm=spec.name,
                               artifact_version=spec.artifact_version,
                               detail={"error": "no_candidate"})
                continue
            # add_shadow_response emits replay.generated for us.
            ctx.store.add_shadow_response(
                e["id"], arm=spec.name, mode="replay", response_text=text,
                model_tag=BASE_MODEL_TAG,
                artifact_version=spec.artifact_version,
            )
            generated += 1
    ctx.results["replay_exchange_ids"] = [e["id"] for e in chat]
    ctx.results["replay_arms"] = [s.name for s in specs]
    note = (f"{generated} candidates over {len(chat)} exchanges, "
            f"arms {[s.name for s in specs]}")
    if dropped:
        note += f", {dropped} older exchange(s) not replayed (cap {REPLAY_MAX_EXCHANGES})"
    if skipped:
        note += f", skipped {sorted(skipped)}"
    return note


def _record_outcomes(ctx: NightlyContext, result, judge_name: str) -> None:
    """Persist one pairwise result's per-prompt verdicts, with provenance."""
    for o in result.outcomes:
        winner = "arm" if o.score > 0.5 else ("base" if o.score < 0.5 else "tie")
        ctx.store.execute(
            "INSERT INTO eval_results (run_id, arm, prompt_id, opponent, winner,"
            " judge, scores, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ctx.results.get("run_id"), result.arm, o.prompt_id, result.opponent,
             winner, judge_name, json.dumps({"score": o.score}), time.time()),
        )
        # Harvested prompt ids are exchange ids, so the verdict lands on the
        # exchange's own trace; curated ids get their own prompt subject.
        subject_type = "exchange" if o.prompt_id.isdigit() else "prompt"
        ctx.store.emit("judge.verdict", subject_type=subject_type,
                       subject_id=o.prompt_id, arm=result.arm,
                       detail={"opponent": result.opponent, "winner": winner,
                               "score": o.score, "judge": judge_name,
                               "category": o.category})


def _eval_split(ctx: NightlyContext, prompts: list, responses_for, arms: list[str],
                judge, split: str) -> list:
    """Judge every arm against base over one split; returns PairwiseResults."""
    from research.eval_runner import run_pairwise

    base_responses = responses_for("base")
    results = []
    for arm in arms:
        result = run_pairwise(prompts, responses_for(arm), base_responses, judge,
                              arm_name=arm, opponent_name="base")
        results.append(result)
        _record_outcomes(ctx, result, judge.name)
        ctx.store.emit("eval.summary", subject_type="run",
                       subject_id=ctx.results.get("run_id"), arm=arm,
                       artifact_version=artifacts.current_version(arm, ctx.artifacts_dir),
                       detail={"split": split, "n": len(result.outcomes),
                               "n_decisive": result.n_decisive,
                               "win_rate": round(result.win_rate, 4),
                               "p_value": round(result.p_value, 4),
                               "judge": judge.name})
    return results


def stage_eval(ctx: NightlyContext) -> str:
    """Judge the replay split nightly, and the curated split on weekly runs.

    The replay split is real prompts the user actually sent, which is what makes
    it worth judging — but its prompts carry no annotations and no category, so
    the pre-registered bar's generic_control condition cannot be measured on it.
    That is measured on the curated split, which has real control probes, and
    which runs weekly because it is ~2x n_prompts x n_arms paid judge calls.
    """
    from research.evalset import EvalPrompt
    from research.judge import FakeJudge, SonnetJudge

    eids = ctx.results.get("replay_exchange_ids", [])
    arms = [a for a in ctx.results.get("replay_arms", []) if a != "base"]
    if not eids or not arms:
        return "nothing to judge"

    def responses_for(arm: str) -> dict[str, str]:
        rows = ctx.store.query(
            "SELECT exchange_id, response_text FROM shadow_responses"
            " WHERE arm = ? AND mode = 'replay' AND exchange_id IN"
            f" ({','.join('?' * len(eids))}) ORDER BY id",
            (arm, *eids),
        )
        return {str(r["exchange_id"]): r["response_text"] for r in rows}

    prompts = []
    for eid in eids:
        e = ctx.store.get_exchange(eid)
        # 'harvested', not 'preference_recall': these are whatever the user
        # happened to say, and labelling them a personalization probe would put
        # a fiction into the results table.
        prompts.append(EvalPrompt(id=str(eid), prompt=e["user_text"],
                                  category="harvested", source="harvested"))

    judge = FakeJudge() if ctx.dry_run else SonnetJudge()
    results = _eval_split(ctx, prompts, responses_for, arms, judge, "replay")
    ctx.results["pairwise"] = results
    note = ", ".join(f"{r.arm}: {r.win_rate:.1%} (n={len(r.outcomes)})" for r in results)

    if ctx.weekly:
        curated = _eval_curated(ctx, arms, judge)
        if curated:
            ctx.results["pairwise_curated"] = curated
            note += " | curated " + ", ".join(
                f"{r.arm}: {r.win_rate:.1%}" for r in curated)
    return note


def _eval_curated(ctx: NightlyContext, arms: list[str], judge) -> list:
    """The weekly curated-probe eval: the only split with real control probes."""
    from research.arms import build_arm_specs
    from research.evalset import load_curated

    try:
        prompts = load_curated()
    except ValueError as e:
        logger.warning("curated split unavailable: %s", e)
        return []

    specs, _ = build_arm_specs(ctx.store, artifacts_dir=ctx.artifacts_dir,
                               arms=arms)
    candidates = {s.name: ctx.candidates_fn(prompts, s) for s in specs}
    if "base" not in candidates:
        return []

    def responses_for(arm: str) -> dict[str, str]:
        return candidates.get(arm, {})

    return _eval_split(ctx, prompts, responses_for, arms, judge, "curated")


def stage_report(ctx: NightlyContext) -> str:
    from research.report import RESULTS_DIR, append_csv_row, write_run_markdown

    results_dir = ctx.results_dir or RESULTS_DIR
    written = []
    for split in ("replay", "curated"):
        key = "pairwise" if split == "replay" else "pairwise_curated"
        results = ctx.results.get(key, [])
        if not results:
            continue
        for r in results:
            append_csv_row(
                r, split=split, date=ctx.date_str,
                run_id=ctx.results.get("run_id"),
                artifact_version=artifacts.current_version(r.arm, ctx.artifacts_dir) or "",
                results_dir=results_dir,
            )
        path = write_run_markdown(
            results, split=split, date=ctx.date_str, results_dir=results_dir,
            protocol_for=(lambda arm: _protocol_summary(arm, results_dir))
            if split == "curated" else None)
        written.append(str(path))
        ctx.store.emit("report.written", subject_type="run",
                       subject_id=ctx.results.get("run_id"),
                       detail={"split": split, "path": str(path),
                               "arms": [r.arm for r in results]})
    if not written:
        return "nothing to report"

    _evaluate_protocol(ctx)
    return ", ".join(written)


def _protocol_summary(arm: str, results_dir: Path) -> str:
    """One-line pre-registered-bar verdict for the markdown digest."""
    from research import protocol

    try:
        rows = protocol.load_rows(Path(results_dir) / "eval.csv")
        return protocol.evaluate_bar(rows, arm=arm).summary()
    except Exception:
        logger.debug("protocol summary unavailable for %s", arm, exc_info=True)
        return ""


def _evaluate_protocol(ctx: NightlyContext) -> None:
    """Record where each arm stands against the pre-registered bar."""
    from research import protocol
    from research.report import RESULTS_DIR

    try:
        rows = protocol.load_rows(Path(ctx.results_dir or RESULTS_DIR) / "eval.csv")
    except Exception:
        logger.debug("protocol evaluation skipped (no results csv)", exc_info=True)
        return
    for arm in {r.arm for r in ctx.results.get("pairwise", [])}:
        try:
            bar = protocol.evaluate_bar(rows, arm=arm)
        except Exception:
            continue
        ctx.store.emit(
            "protocol.evaluated", subject_type="artifact",
            subject_id=f"{arm}/{artifacts.current_version(arm, ctx.artifacts_dir) or '-'}",
            arm=arm,
            detail={"improved": bar.improved,
                    "conditions": {c.number: c.passed for c in bar.conditions},
                    "summary": bar.summary()},
        )


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
                results_dir: Optional[Path] = None,
                stages: Optional[list[str]] = None,
                weekly: Optional[bool] = None) -> dict[str, str]:
    """Run the loop; returns {stage: status}. Never raises out of a stage."""
    now = datetime.now()
    ctx = NightlyContext(
        store=store,
        date_str=date_str or now.strftime("%Y%m%d"),
        since_ts=time.time() - LOOKBACK_HOURS * 3600,
        dry_run=dry_run,
        # Sunday carries the weekly curated eval, which is what the protocol's
        # "two consecutive weekly evals" condition is defined over.
        weekly=now.weekday() == 6 if weekly is None else weekly,
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
    )
    run_id = store.execute(
        "INSERT INTO runs (started_ts, stage_status) VALUES (?, '{}')", (time.time(),))
    ctx.results["run_id"] = run_id
    store.set_run_context(run_id, "run")
    store.emit("run.started", subject_type="run", subject_id=run_id,
               detail={"dry_run": dry_run, "weekly": ctx.weekly,
                       "stages": stages, "date": ctx.date_str})

    status: dict[str, str] = {}
    try:
        for name, fn in STAGES:
            if stages is not None and name not in stages:
                status[name] = "skipped (not selected)"
                continue
            store.set_run_context(run_id, name)
            t0 = time.monotonic()
            try:
                note = fn(ctx)
                elapsed = time.monotonic() - t0
                status[name] = f"ok ({elapsed:.0f}s): {note}"
                store.emit("run.stage_ok", subject_type="run", subject_id=run_id,
                           detail={"stage": name, "seconds": round(elapsed, 1),
                                   "note": note[:300]})
            except Exception as e:
                logger.exception("nightly stage %s failed", name)
                status[name] = f"FAILED: {type(e).__name__}: {e}"
                store.emit("run.stage_failed", subject_type="run", subject_id=run_id,
                           detail={"stage": name, "error": type(e).__name__,
                                   "message": str(e)[:300]})
            store.execute(
                "UPDATE runs SET stage_status = ?, finished_ts = ? WHERE id = ?",
                (json.dumps(status), time.time(), run_id),
            )
    finally:
        store.set_run_context(run_id, "run")
        store.emit("run.finished", subject_type="run", subject_id=run_id,
                   detail={"failed": [k for k, v in status.items()
                                      if v.startswith("FAILED")]})
        store.set_run_context(None, "live")
    return status
