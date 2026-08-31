"""Research CLI: python -m research {status,doctor,harvest,eval,rate,trace,nightly,protocol,revert}

Standalone of the running assistant — operates directly on research.db and the
artifact directories. The nightly orchestrator (research/nightly.py) drives the
same functions on a schedule.
"""

from __future__ import annotations

import argparse
import time

from research.db import ResearchStore


def _artifacts_dir(args: argparse.Namespace):
    """The artifacts root for this invocation (--artifacts-dir, else default)."""
    from pathlib import Path

    from research import artifacts

    return (Path(args.artifacts_dir).expanduser() if args.artifacts_dir
            else artifacts.DEFAULT_ARTIFACTS_DIR)


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_detail(raw: str | None) -> str:
    """Compact one-line rendering of an event's detail JSON."""
    import json

    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return str(raw)
    if not isinstance(data, dict):
        return str(data)
    parts = []
    for k, v in data.items():
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, list) and len(v) > 6:
            parts.append(f"{k}=[{len(v)} items]")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def cmd_status(args: argparse.Namespace) -> int:
    import json

    from research import artifacts

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

    art_dir = _artifacts_dir(args)
    print("artifacts:")
    # The classic arms first, then anything else discovered on disk — a new
    # arm shows up here (and in self_status/doctor) with no code change.
    arms = ["memory", "lora", "prompt"]
    if art_dir.exists():
        from introspection.providers import discover_arms

        arms += [a for a in discover_arms(art_dir) if a not in arms]
    for arm in arms:
        current = artifacts.current_version(arm, art_dir)
        n_versions = len(artifacts.list_versions(arm, art_dir))
        print(f"  {arm:8} current={current or '-'} ({n_versions} version(s))")

    rows = store.query(
        "SELECT started_ts, finished_ts, stage_status FROM runs ORDER BY id DESC LIMIT 1"
    )
    if rows:
        row = rows[0]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["started_ts"]))
        print(f"last nightly run ({when}):")
        for stage, note in json.loads(row["stage_status"] or "{}").items():
            print(f"  {stage:8} {note}")

    events = store.recent_events(limit=10)
    if events:
        print("recent events:")
        for e in reversed(events):
            print(f"  {_fmt_ts(e['ts'])} {e['stage']:8} {e['event']:26}"
                  f" {e['subject_type']} {e['subject_id']}")
    if store.emit_failures:
        print(f"WARNING: {store.emit_failures} event emit failure(s) this session")

    # Placeholder probes silently corrupt the judge rubric, so keep the chore
    # visible until they're written.
    try:
        from research.evalset import count_placeholders

        n_todo = count_placeholders()
        if n_todo:
            print(f"eval set: {n_todo} curated probe(s) still marked FILL-IN "
                  f"(curated split will refuse to run)")
        from research.evalset import count_drafts

        n_drafts = count_drafts()
        if n_drafts:
            print(f"eval set: {n_drafts} draft probe(s) awaiting your "
                  f"annotations (research/data/evalset/curated.yaml)")
    except Exception:
        pass
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Self-diagnosis check suite — the same checks Friday runs when asked to
    'run a self-diagnosis' (workflows/introspection.py). Exit 1 on any FAIL."""
    import os
    from pathlib import Path

    from introspection import CheckStatus, Paths, format_report, run_doctor

    db_path = Path(args.db).expanduser()
    state_dir = db_path.parent
    paths = Paths(
        state_dir=state_dir,
        research_db=db_path,
        artifacts_dir=_artifacts_dir(args),
        audit_db=Path(os.getenv("FRIDAY_AUDIT_DB",
                                state_dir / "audit.db")).expanduser(),
    )
    results = run_doctor(paths)
    print("self-diagnosis:")
    print(format_report(results))
    return 1 if any(r.status is CheckStatus.FAIL for r in results) else 0


def cmd_trace(args: argparse.Namespace) -> int:
    """Show the provenance trail: what the loop used to improve itself, and when."""
    import json

    from research import artifacts, provenance

    store = ResearchStore(args.db)

    if args.exchange is not None:
        events = store.events_for("exchange", args.exchange)
        exchange = store.get_exchange(args.exchange)
        if exchange is None and not events:
            print(f"no such exchange: {args.exchange}")
            return 1
        if args.json:
            print(json.dumps({"exchange": exchange, "events": events},
                             indent=2, default=str))
            return 0
        if exchange is not None:
            print(f"exchange {exchange['id']}   {_fmt_ts(exchange['ts'])}  "
                  f"{exchange.get('channel') or '-'}  route={exchange.get('route')}  "
                  f"model={exchange.get('model') or '-'}")
            print(f"  user:  {exchange['user_text'][:160]!r}")
            print(f"  reply: {exchange['reply_text'][:160]!r}\n")
        _print_events(events)
        return 0

    if args.artifact:
        art_dir = _artifacts_dir(args)
        arm, _, version = args.artifact.partition("/")
        if not version:
            version = artifacts.current_version(arm, art_dir) or ""
            if not version:
                print(f"arm {arm!r} has no current version")
                return 1
        manifest = provenance.read_manifest(arm, version, art_dir)
        events = store.events_for_artifact(f"{arm}/{version}")
        if args.json:
            print(json.dumps({"manifest": manifest, "events": events},
                             indent=2, default=str))
            return 0
        current = artifacts.current_version(arm, art_dir)
        print(f"artifact {arm}/{version}"
              f"{'   (current)' if current == version else ''}")
        if manifest is None:
            print("  no provenance manifest (built before provenance landed)")
        else:
            print("\n  inputs (provenance.json)")
            for key, value in sorted(manifest.get("inputs", {}).items()):
                shown = f"{len(value)} {_summarize_ids(value)}" if isinstance(value, list) else value
                print(f"    {key:28} {shown}")
            for section in ("dataset", "params"):
                if manifest.get(section):
                    print(f"\n  {section}")
                    for key, value in sorted(manifest[section].items()):
                        print(f"    {key:28} {value}")
        print("\n  timeline")
        _print_events(events, indent="    ")
        return 0

    if args.run is not None:
        events = store.events_for_run(args.run)
        if args.json:
            print(json.dumps(events, indent=2, default=str))
            return 0
        print(f"run {args.run}")
        _print_events(events)
        return 0

    since = 0.0
    if args.since:
        since = time.mktime(time.strptime(args.since, "%Y-%m-%d"))
    events = store.recent_events(limit=args.limit, event=args.event, arm=args.arm,
                                 since_ts=since)
    if args.json:
        print(json.dumps(events, indent=2, default=str))
        return 0
    _print_events(list(reversed(events)))
    return 0


def _summarize_ids(values: list) -> str:
    """'ids 3..431' for a long id list, else the list itself."""
    if not values:
        return ""
    if len(values) <= 6:
        return str(values)
    numeric = [v for v in values if isinstance(v, int)]
    if len(numeric) == len(values):
        return f"(ids {min(numeric)}..{max(numeric)})"
    return f"({len(values)} items)"


def _print_events(events: list[dict], indent: str = "  ") -> None:
    if not events:
        print(f"{indent}(no events)")
        return
    for e in events:
        # arm is a column, not part of detail, but "which arm did this" is the
        # whole point of a replay/judge line.
        arm = f"arm={e['arm']} " if e["arm"] else ""
        artifact = f"  [{e['artifact_version']}]" if e["artifact_version"] else ""
        detail = _fmt_detail(e["detail"])
        print(f"{indent}{_fmt_ts(e['ts'])}  {e['stage']:8} {e['event']:26}"
              f" {arm}{detail}{artifact}")


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
    from research.arms import build_arm_specs
    from research.eval_runner import run_pairwise
    from research.evalset import load_curated, load_harvested
    from research.generate import generate_candidates
    from research.judge import FakeJudge, LocalJudge, SonnetJudge
    from research.report import append_csv_row, write_run_markdown

    store = ResearchStore(args.db)
    if args.split in ("curated", "reserve"):
        try:
            prompts = load_curated(allow_placeholders=args.allow_placeholders,
                                   reserve_only=args.split == "reserve")
        except ValueError as e:
            print(f"{e}\n\nRe-run with --allow-placeholders for a smoke run "
                  f"(results will be noise on those probes).")
            return 1
    else:
        prompts = load_harvested(store, after_ts=args.after_ts)
    if not prompts:
        print("no eval prompts in split")
        return 1

    if args.judge == "sonnet":
        judge = SonnetJudge()
        print(f"judge: {judge.name} (PAID API — ~{2 * len(prompts)} calls per arm)")
    elif args.judge == "local":
        judge = LocalJudge()
    else:
        judge = FakeJudge()

    requested = [a.strip() for a in args.arms.split(",")
                 if a.strip() and a.strip() != "base"]
    specs, skipped = build_arm_specs(store, arms=requested)
    for arm, reason in sorted(skipped.items()):
        print(f"arm {arm!r} skipped: {reason}")
    by_name = {s.name: s for s in specs}
    arms = [s.name for s in specs if s.name != "base"]
    if not arms:
        print("no arms to evaluate")
        return 1

    base_responses = generate_candidates(prompts, by_name["base"])

    results = []
    for arm in arms:
        arm_responses = generate_candidates(prompts, by_name[arm])
        result = run_pairwise(prompts, arm_responses, base_responses, judge,
                              arm_name=arm, opponent_name="base")
        results.append(result)
        append_csv_row(result, split=args.split,
                       artifact_version=by_name[arm].artifact_version or "")
        print(f"{arm} vs base: win rate {result.win_rate:.1%} "
              f"(n={len(result.outcomes)}, p={result.p_value:.3f})")
    if results:
        path = write_run_markdown(results, split=args.split)
        print(f"report: {path}")
    return 0


def cmd_protocol(args: argparse.Namespace) -> int:
    """Where each arm stands against the pre-registered bar."""
    from research import protocol

    try:
        rows = protocol.load_rows()
    except FileNotFoundError:
        print("no results/eval.csv yet — run an eval first")
        return 1

    arms = [args.arm] if args.arm else sorted(
        {r["arm"] for r in rows if r.get("split") == args.split})
    if not arms:
        print(f"no {args.split} rows in results/eval.csv")
        return 1

    for arm in arms:
        bar = protocol.evaluate_bar(rows, arm=arm, split=args.split)
        print(f"\n{arm}: {bar.summary()}")
        for c in bar.conditions:
            mark = {True: "PASS", False: "FAIL", None: "PENDING"}[c.passed]
            print(f"  {c.number}. {mark:8} {c.name:34} {c.detail}")

    if getattr(args, "pooled", False):
        print("\nPooled primary endpoint "
              f"(last {protocol.POOL_WINDOW} weekly evals, Holm over study arms):")
        pooled = {arm: protocol.evaluate_pooled(rows, arm=arm, split=args.split)
                  for arm in arms}
        adjusted = protocol.holm({a: p.p_value for a, p in pooled.items()
                                  if a in protocol.STUDY_ARMS_FOR_HOLM and p.dates})
        for arm in arms:
            p = pooled[arm]
            line = f"  {arm}: {p.summary()}"
            if arm in adjusted:
                line += f", Holm-adjusted p={adjusted[arm]:.4f}"
            print(line)
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    """Human anchor: rate anonymized response pairs.

    --mode shadow (default): production reply vs live shadow reply.
    --mode arm-base: the pre-registered anchor — arm vs base pairs from the
    most recent judged eval run, so judge-human agreement is computed on the
    same tournament the judge scored.

    Presentation order is randomized per pair and the mapping recorded, so
    these ratings can calibrate the LLM judges. Aggregates only in the repo;
    the texts stay in research.db.
    """
    import random
    import time as _time

    if getattr(args, "mode", "shadow") == "arm-base":
        return _rate_arm_base(args)

    store = ResearchStore(args.db)
    rows = store.query(
        "SELECT e.id, e.user_text, e.reply_text, s.response_text, s.model_tag"
        " FROM exchanges e JOIN shadow_responses s ON s.exchange_id = e.id"
        " WHERE s.arm = 'base' AND s.mode = 'live'"
        " AND e.id NOT IN (SELECT CAST(prompt_id AS INTEGER) FROM eval_results WHERE judge = 'human')"
        " ORDER BY e.ts DESC LIMIT ?",
        (args.pairs,),
    )
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
        store.execute(
            "INSERT INTO eval_results (run_id, arm, prompt_id, opponent, winner,"
            " judge, scores, ts) VALUES (NULL, 'shadow-live', ?, 'production', ?,"
            " 'human', NULL, ?)",
            (str(row["id"]), winner, _time.time()),
        )
        store.emit("judge.verdict", subject_type="exchange", subject_id=row["id"],
                   arm="shadow-live",
                   detail={"judge": "human", "opponent": "production",
                           "winner": winner})
        rated += 1
    print(f"\nrated {rated} pair(s)")
    return 0


def _pair_texts(store, run_id: int, arm: str, prompt_id: str,
                curated_prompts: dict):
    """(user prompt, arm response, base response) for one judged pair, or None."""
    if prompt_id.isdigit():
        e = store.get_exchange(int(prompt_id))
        if not e:
            return None
        texts = {}
        for side in (arm, "base"):
            rows = store.query(
                "SELECT response_text FROM shadow_responses WHERE exchange_id = ?"
                " AND arm = ? AND mode = 'replay' ORDER BY id DESC LIMIT 1",
                (int(prompt_id), side))
            if not rows:
                return None
            texts[side] = rows[0]["response_text"]
        return e["user_text"], texts[arm], texts["base"]
    prompt_text = curated_prompts.get(prompt_id)
    if not prompt_text:
        return None
    texts = {}
    for side in (arm, "base"):
        rows = store.query(
            "SELECT response_text FROM curated_responses WHERE run_id = ?"
            " AND prompt_id = ? AND arm = ? ORDER BY id DESC LIMIT 1",
            (run_id, prompt_id, side))
        if not rows:
            return None
        texts[side] = rows[0]["response_text"]
    return prompt_text, texts[arm], texts["base"]


def _rate_arm_base(args: argparse.Namespace) -> int:
    """Rate arm-vs-base pairs from the most recent judged run; report
    judge-human direction agreement and append it to that run's digest."""
    import random
    import time as _time
    from datetime import datetime

    store = ResearchStore(args.db)
    runs = store.query(
        "SELECT run_id FROM eval_results WHERE judge LIKE 'sonnet%'"
        " AND run_id IS NOT NULL ORDER BY id DESC LIMIT 1")
    if not runs:
        print("no judged eval runs yet — nothing to rate")
        return 1
    run_id = runs[0]["run_id"]

    sonnet = {(r["arm"], r["prompt_id"]): r["winner"] for r in store.query(
        "SELECT arm, prompt_id, winner FROM eval_results"
        " WHERE run_id = ? AND judge LIKE 'sonnet%'", (run_id,))}
    done = {(r["arm"], r["prompt_id"]) for r in store.query(
        "SELECT arm, prompt_id FROM eval_results"
        " WHERE run_id = ? AND judge = 'human'", (run_id,))}
    candidates = sorted(set(sonnet) - done)
    if not candidates:
        print(f"every pair from run {run_id} is already rated")
        return 0

    curated_prompts = {}
    try:
        from research.evalset import load_curated
        curated_prompts = {p.id: p.prompt for p in load_curated()}
    except Exception:
        pass

    random.Random(run_id).shuffle(candidates)
    rated = 0
    agree = decisive = 0
    for arm, prompt_id in candidates:
        if rated >= args.pairs:
            break
        pair = _pair_texts(store, run_id, arm, prompt_id, curated_prompts)
        if pair is None:
            continue
        prompt_text, arm_text, base_text = pair
        rng = random.Random(f"{run_id}:{arm}:{prompt_id}")
        first_is_arm = rng.random() < 0.5
        first, second = (arm_text, base_text) if first_is_arm else (base_text, arm_text)
        print(f"\n--- pair {rated + 1}/{min(args.pairs, len(candidates))} ---")
        print(f"PROMPT: {prompt_text}\n")
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
            winner = "arm" if picked_first == first_is_arm else "base"
        store.execute(
            "INSERT INTO eval_results (run_id, arm, prompt_id, opponent, winner,"
            " judge, scores, ts) VALUES (?, ?, ?, 'base', ?, 'human', NULL, ?)",
            (run_id, arm, prompt_id, winner, _time.time()),
        )
        subject_type = "exchange" if prompt_id.isdigit() else "prompt"
        store.emit("judge.verdict", subject_type=subject_type, subject_id=prompt_id,
                   arm=arm, detail={"judge": "human", "opponent": "base",
                                    "winner": winner, "run_id": run_id})
        rated += 1
        s_winner = sonnet.get((arm, prompt_id))
        if winner != "tie" and s_winner in ("arm", "base"):
            decisive += 1
            agree += winner == s_winner

    print(f"\nrated {rated} pair(s) from run {run_id}")
    if decisive:
        rate = agree / decisive
        line = (f"Human anchor (run {run_id}): {rated} pairs rated, "
                f"judge-human agreement {rate:.0%} over {decisive} decisive.")
        print(line)
        try:
            from research.report import RESULTS_DIR
            row = store.query("SELECT started_ts FROM runs WHERE id = ?", (run_id,))
            date_str = datetime.fromtimestamp(row[0]["started_ts"]).strftime("%Y%m%d")
            digest = RESULTS_DIR / "nightly" / f"{date_str}.md"
            if digest.exists():
                digest.write_text(digest.read_text().rstrip() + f"\n\n{line}\n")
                print(f"appended to {digest}")
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "could not append human-anchor line to digest", exc_info=True)
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    """Emit paper-ready tables and figures into results/paper/."""
    from pathlib import Path

    from research.paper import build_paper_outputs
    from research.report import RESULTS_DIR

    out_dir = Path(args.out) if args.out else RESULTS_DIR / "paper"
    store = ResearchStore(args.db)
    try:
        written = build_paper_outputs(out_dir=out_dir, store=store)
    except FileNotFoundError:
        print("no results/eval.csv yet — run an eval first")
        return 1
    for p in written:
        print(f"wrote {p}")
    return 0


def cmd_nightly(args: argparse.Namespace) -> int:
    import fcntl
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from research.nightly import run_nightly

    # One nightly at a time: an overrun (slow training) must not double up
    # with the next launchd firing. flock releases automatically on exit.
    lock_path = Path("~/.friday/research/nightly.lock").expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another nightly run is still in progress — exiting")
        return 0

    store = ResearchStore(args.db)
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    status = run_nightly(store, dry_run=args.dry_run, stages=stages,
                         weekly=args.weekly, artifacts_dir=_artifacts_dir(args))
    print("\nnightly run:")
    failed = False
    for stage, note in status.items():
        print(f"  {stage:8} {note}")
        failed = failed or note.startswith("FAILED")
    if failed and not args.dry_run:
        # launchd discards this exit code, so a broken loop used to be
        # invisible until someone ran `research status`. Best-effort ping;
        # a silent no-op without a Telegram token.
        from introspection.alerts import format_nightly_alert, send_telegram

        message = format_nightly_alert(status)
        if message and send_telegram(message):
            print("failure alert sent to Telegram")

    # Weekly human-anchor nudge (PROTOCOL.md): the anchor only exists if the
    # user actually rates pairs, so remind them right after the Sunday eval.
    from datetime import datetime as _dt
    weekly = args.weekly if args.weekly is not None else _dt.now().weekday() == 6
    if weekly and not args.dry_run and not status.get("eval", "").startswith("FAILED"):
        try:
            from introspection.alerts import send_telegram
            send_telegram("Weekly eval judged. Please rate this week's pairs: "
                          "python -m research rate --mode arm-base (~20 pairs, ~10 min).")
        except Exception:
            pass
    return 1 if failed else 0


def cmd_revert(args: argparse.Namespace) -> int:
    # Shared with the self_repair workflow so a revert is one code path and
    # one artifact.advanced event, whoever asks for it.
    from research.ops import revert_arm

    try:
        revert_arm(args.arm, args.to, artifacts_dir=_artifacts_dir(args),
                   db_path=args.db, via="manual revert")
    except ValueError as e:
        print(e)
        return 1
    print(f"{args.arm}: current -> {args.to}")
    return 0


def main(argv=None) -> int:
    # Same as main.py: ANTHROPIC_API_KEY lives in ./.env, and launchd (the
    # nightly job) sources no shell profile, so the evolver and the Sonnet
    # judge would otherwise run unauthenticated at 03:30.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(prog="python -m research")
    parser.add_argument("--db", default="~/.friday/research.db")
    parser.add_argument("--artifacts-dir", default=None,
                        help="default ~/.friday/research; must pair with --db")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="DB counts and recent activity")

    sub.add_parser("doctor", help="self-diagnosis check suite (exit 1 on any FAIL)")

    p_harvest = sub.add_parser("harvest", help="mine implicit feedback signals")
    p_harvest.add_argument("--hours", type=float, default=36.0,
                           help="look-back window (default 36h for overlap)")

    p_eval = sub.add_parser("eval", help="pairwise-judge arms vs base")
    p_eval.add_argument("--arms", default="base", help="comma-separated arm names")
    p_eval.add_argument("--split", choices=("curated", "harvested", "reserve"),
                        default="curated",
                        help="reserve: the held-out probes, for the final "
                             "paper numbers only")
    p_eval.add_argument("--judge", choices=("fake", "local", "sonnet"), default="fake")
    p_eval.add_argument("--after-ts", type=float, default=0.0,
                        help="harvested split: only prompts newer than this epoch ts")
    p_eval.add_argument("--allow-placeholders", action="store_true",
                        help="curated split: run despite FILL-IN annotations (noise)")

    p_rate = sub.add_parser("rate", help="human-rate anonymized response pairs")
    p_rate.add_argument("--pairs", type=int, default=20)
    p_rate.add_argument("--mode", choices=("shadow", "arm-base"), default="shadow",
                        help="shadow: production vs live shadow; arm-base: the "
                             "pre-registered anchor over the latest eval run")

    p_trace = sub.add_parser(
        "trace", help="provenance: what the loop used to improve itself, and when")
    p_trace.add_argument("--exchange", type=int, help="one turn's full life story")
    p_trace.add_argument("--artifact", help="'<arm>/<version>', or just '<arm>' for current")
    p_trace.add_argument("--run", type=int, help="all events from one nightly run")
    p_trace.add_argument("--since", help="YYYY-MM-DD lower bound for the event feed")
    p_trace.add_argument("--event", help="filter the feed to one event name")
    p_trace.add_argument("--arm", help="filter the feed to one arm")
    p_trace.add_argument("--limit", type=int, default=50)
    p_trace.add_argument("--json", action="store_true", help="machine-readable output")

    p_nightly = sub.add_parser("nightly", help="run the nightly learning loop")
    p_nightly.add_argument("--dry-run", action="store_true",
                           help="FakeJudge, fake generation, no training, no paid calls")
    p_nightly.add_argument("--stages", default=None,
                           help="comma-separated subset of stages to run")
    p_nightly.add_argument("--weekly", action="store_true", default=None,
                           help="also judge the curated split (automatic on Sundays)")

    p_protocol = sub.add_parser(
        "protocol", help="where each arm stands against the pre-registered bar")
    p_protocol.add_argument("--arm", default=None, help="default: every arm with rows")
    p_protocol.add_argument("--pooled", action="store_true",
                            help="also print the pooled primary endpoint with "
                                 "Wilson CIs and Holm-adjusted p-values")
    p_protocol.add_argument("--split", default="curated",
                            choices=("curated", "replay", "harvested"))

    p_revert = sub.add_parser("revert", help="repoint an arm's current artifact")
    p_revert.add_argument("--arm", required=True,
                          help="arm name (memory, lora, prompt, or any arm on disk)")
    p_revert.add_argument("--to", required=True, help="version name, e.g. v20260718")

    p_paper = sub.add_parser(
        "paper", help="paper-ready tables and figures from the accumulated results")
    p_paper.add_argument("--out", default=None,
                         help="output dir (default: results/paper)")

    args = parser.parse_args(argv)
    return {"status": cmd_status, "doctor": cmd_doctor, "harvest": cmd_harvest,
            "eval": cmd_eval, "rate": cmd_rate, "trace": cmd_trace,
            "nightly": cmd_nightly, "protocol": cmd_protocol,
            "revert": cmd_revert, "paper": cmd_paper}[args.command](args)
