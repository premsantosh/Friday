"""Research CLI: python -m research {status,harvest,eval}

Standalone of the running assistant — operates directly on research.db and the
artifact directories. The nightly orchestrator (research/nightly.py) drives the
same functions on a schedule.
"""

from __future__ import annotations

import argparse
import time

from research.db import ResearchStore


def cmd_status(args: argparse.Namespace) -> int:
    store = ResearchStore(args.db)
    counts = store.counts()
    print("research.db counts:")
    for table, n in counts.items():
        print(f"  {table:18} {n}")
    day_ago = time.time() - 86400
    recent = store.exchanges_since(day_ago)
    routes: dict[str, int] = {}
    for e in recent:
        routes[e["route"] or "unknown"] = routes.get(e["route"] or "unknown", 0) + 1
    print(f"last 24h: {len(recent)} exchanges "
          f"({', '.join(f'{k}={v}' for k, v in sorted(routes.items())) or 'none'})")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    """Mine implicit feedback from recent exchanges into the feedback table."""
    from research.miners import mine_all

    store = ResearchStore(args.db)
    since = time.time() - args.hours * 3600
    exchanges = store.exchanges_since(since)
    signals = mine_all(exchanges)
    inserted = 0
    for sig in signals:
        if store.has_feedback(sig.exchange_id, sig.source):
            continue  # idempotent across repeated harvests
        store.add_feedback(sig.exchange_id, kind="implicit", signal=sig.signal,
                           source=sig.source, details=sig.details)
        inserted += 1
    print(f"harvest: {len(exchanges)} exchanges in window, "
          f"{len(signals)} signals mined, {inserted} new")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Generate candidates for the requested arms and judge them vs base."""
    from research.eval_runner import run_pairwise
    from research.evalset import load_curated, load_harvested
    from research.generate import ArmSpec, generate_candidates
    from research.judge import FakeJudge, QwenJudge, SonnetJudge
    from research.report import append_csv_row, write_run_markdown

    store = ResearchStore(args.db)
    if args.split == "curated":
        prompts = load_curated()
    else:
        prompts = load_harvested(store, after_ts=args.after_ts)
    if not prompts:
        print("no eval prompts in split")
        return 1

    if args.judge == "sonnet":
        judge = SonnetJudge()
        print(f"judge: {judge.name} (PAID API — ~{2 * len(prompts)} calls per arm)")
    elif args.judge == "qwen":
        judge = QwenJudge()
    else:
        judge = FakeJudge()

    arms = [a.strip() for a in args.arms.split(",") if a.strip() and a.strip() != "base"]
    # Arms beyond vanilla base attach their artifact; artifact loaders arrive
    # with M3-M5 — until then only 'base' self-play smoke runs are possible.
    specs = {"base": ArmSpec(name="base")}
    for arm in arms:
        if arm not in specs:
            print(f"arm {arm!r} has no artifact loader yet — skipping")
    base_responses = generate_candidates(prompts, specs["base"])

    results = []
    for arm in arms:
        if arm not in specs:
            continue
        arm_responses = generate_candidates(prompts, specs[arm])
        result = run_pairwise(prompts, arm_responses, base_responses, judge,
                              arm_name=arm, opponent_name="base")
        results.append(result)
        append_csv_row(result, split=args.split)
        print(f"{arm} vs base: win rate {result.win_rate:.1%} "
              f"(n={len(result.outcomes)}, p={result.p_value:.3f})")
    if results:
        path = write_run_markdown(results, split=args.split)
        print(f"report: {path}")
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    """Human anchor: rate real (production reply vs shadow reply) pairs.

    Presentation order is randomized per pair and the mapping recorded, so
    these ratings can calibrate the LLM judges. Aggregates only in the repo;
    the texts stay in research.db.
    """
    import random
    import time as _time

    store = ResearchStore(args.db)
    rows = store.conn.execute(
        "SELECT e.id, e.user_text, e.reply_text, s.response_text, s.model_tag"
        " FROM exchanges e JOIN shadow_responses s ON s.exchange_id = e.id"
        " WHERE s.arm = 'base' AND s.mode = 'live'"
        " AND e.id NOT IN (SELECT CAST(prompt_id AS INTEGER) FROM eval_results WHERE judge = 'human')"
        " ORDER BY e.ts DESC LIMIT ?",
        (args.pairs,),
    ).fetchall()
    if not rows:
        print("no unrated (production, shadow) pairs — run with FRIDAY_RESEARCH=1 first")
        return 1

    rated = 0
    for row in rows:
        rng = random.Random(row["id"])
        first_is_production = rng.random() < 0.5
        first, second = ((row["reply_text"], row["response_text"])
                         if first_is_production
                         else (row["response_text"], row["reply_text"]))
        print(f"\n--- pair {rated + 1}/{len(rows)} (exchange {row['id']}) ---")
        print(f"YOU SAID: {row['user_text']}\n")
        print(f"  [1] {first}\n")
        print(f"  [2] {second}\n")
        choice = input("Better for you? 1 / 2 / t(ie) / s(kip) / q(uit): ").strip().lower()
        if choice == "q":
            break
        if choice == "s" or choice not in ("1", "2", "t"):
            continue
        if choice == "t":
            winner = "tie"
        else:
            picked_first = choice == "1"
            winner = ("production" if picked_first == first_is_production else "shadow")
        with store._lock:
            store.conn.execute(
                "INSERT INTO eval_results (run_id, arm, prompt_id, opponent, winner,"
                " judge, scores, ts) VALUES (NULL, 'shadow-live', ?, 'production', ?,"
                " 'human', NULL, ?)",
                (str(row["id"]), winner, _time.time()),
            )
            store.conn.commit()
        rated += 1
    print(f"\nrated {rated} pair(s)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research")
    parser.add_argument("--db", default="~/.friday/research.db")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="DB counts and recent activity")

    p_harvest = sub.add_parser("harvest", help="mine implicit feedback signals")
    p_harvest.add_argument("--hours", type=float, default=36.0,
                           help="look-back window (default 36h for overlap)")

    p_eval = sub.add_parser("eval", help="pairwise-judge arms vs base")
    p_eval.add_argument("--arms", default="base", help="comma-separated arm names")
    p_eval.add_argument("--split", choices=("curated", "harvested"), default="curated")
    p_eval.add_argument("--judge", choices=("fake", "qwen", "sonnet"), default="fake")
    p_eval.add_argument("--after-ts", type=float, default=0.0,
                        help="harvested split: only prompts newer than this epoch ts")

    p_rate = sub.add_parser("rate", help="human-rate production vs shadow pairs")
    p_rate.add_argument("--pairs", type=int, default=20)

    args = parser.parse_args(argv)
    return {"status": cmd_status, "harvest": cmd_harvest, "eval": cmd_eval,
            "rate": cmd_rate}[args.command](args)
