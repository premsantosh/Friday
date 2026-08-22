"""
Standing-watch CLI — create, inspect and control TableCheck watches from a
terminal, against the same session store the running app uses.

    python -m workflows.reservations.watch_cli add \\
        "Watch https://www.tablecheck.com/en/benfiddich-tokyo/reserve BenFiddich \\
         for September 5th, 9th, 10th or 11th for 4 and book it" \\
        [--user 123456789] [--release 2026-08-20T00:00+09:00 --burst-until 2026-08-21T00:00+09:00]
    python -m workflows.reservations.watch_cli list [--user …]
    python -m workflows.reservations.watch_cli say "any luck?" [--user …]   # any control verb
    python -m workflows.reservations.watch_cli probe benfiddich-tokyo 4 2026-09  # live read-only check

`add` runs the exact conversational flow ("Watch … → yes") so the watch is
indistinguishable from one set up over Telegram; `--user` defaults to the
Telegram notify chat id so Telegram control verbs ("stop watching", "any luck",
"resume booking") reach it. The watch only *advances* while `python main.py`
(or `--telegram`) is running — this tool never polls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv


def _store():
    from core.conversation import SqliteSessionStore
    path = os.path.expanduser(os.getenv("FRIDAY_SESSION_STORE", "~/.friday/sessions.db"))
    return SqliteSessionStore(path)


def _manager():
    from core.conversation import SessionManager
    from workflows.base import WorkflowManager
    from workflows.reservations import ReservationWorkflow
    store = _store()
    wf = ReservationWorkflow(session_store=store)
    wm = WorkflowManager()
    wm.register(wf)
    return SessionManager(store, wm, default_timeout_s=1800), wf, store


def _default_user() -> str:
    uid = os.getenv("TELEGRAM_NOTIFY_CHAT_ID")
    if not uid:
        allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        uid = next((c.strip() for c in allowed.split(",") if c.strip()), "default")
    return uid


def _parse_when(text: str) -> float:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise SystemExit(f"--release/--burst-until need a timezone offset: {text!r}")
    return dt.timestamp()


async def cmd_add(args) -> int:
    mgr, wf, store = _manager()
    turn = await mgr.open(wf, args.text, {}, args.user)
    print(f"Friday: {turn.message}")
    if turn.control.name != "AWAIT_CONFIRMATION":
        print("(no confirmation requested — nothing armed)")
        return 1
    if args.dry_run:
        mgr.cancel(args.user, "dry run")
        print("(dry run — declined)")
        return 0
    turn = await mgr.handle(args.user, "yes")
    print(f"Friday: {turn.message}")
    watches = [s for s in store.list_waiting()
               if s.user_id == args.user and s.slots.get("watchlist_state")]
    if not watches:
        print("No WAITING watch session found after confirmation.")
        return 1
    session = max(watches, key=lambda s: s.created_at)
    ws = session.slots["watchlist_state"]
    if args.release:
        from workflows.reservations.snipe import describe_fire, resolve_timezone
        fire = _parse_when(args.release)
        until = _parse_when(args.burst_until) if args.burst_until else fire + wf._tc_burst_tail_s
        tz = resolve_timezone(ws.get("venue_tz"))
        ws["release"] = {"checked": True, "fire_ts": fire, "burst_until": until,
                         "display": describe_fire(fire, tz), "source": "you",
                         "confidence": 1.0, "quote": ""}
        session.wake_at = time.time() + 2
        store.save(session)
        print(f"Release window set: {ws['release']['display']} → bursting until "
              f"{describe_fire(until, tz)}")
    print(f"Watch session {session.session_id} armed for user {args.user}.")
    return 0


def cmd_list(args) -> int:
    store = _store()
    now = time.time()
    rows = [s for s in store.list_waiting() if s.slots.get("watchlist_state")
            and (not args.user or s.user_id == args.user)]
    if not rows:
        print("No standing watches.")
        return 0
    for s in rows:
        ws = s.slots["watchlist_state"]
        print(f"- {ws.get('business_name')} ({ws.get('slug')}) user={s.user_id} "
              f"session={s.session_id} state={s.fsm_state}")
        for c in ws.get("criteria", []):
            print(f"    {c['date_start']}..{c['date_end']} x{c['party_size']} "
                  f"times={c.get('times')} [{c.get('status')}]")
        rel = ws.get("release") or {}
        if rel.get("fire_ts"):
            print(f"    release: {rel.get('display')} (burst until "
                  f"{datetime.fromtimestamp(rel['burst_until']).isoformat(timespec='minutes')})")
        ab = ws.get("autobook") or {}
        if ab.get("plan"):
            print(f"    autobook: attempts={ab.get('count', 0)} paused={ab.get('paused')} "
                  f"booked={ab.get('booked')}")
        wake = s.wake_at
        print(f"    failures={ws.get('failures', 0)} paused={ws.get('paused')} "
              f"last_checked={ws.get('last_checked_at') and int(now - ws['last_checked_at'])}s ago "
              f"next_wake_in={wake and int(wake - now)}s")
    return 0


async def cmd_say(args) -> int:
    mgr, wf, store = _manager()
    turn = await mgr.open(wf, args.text, {}, args.user)
    print(f"Friday: {turn.message}")
    return 0


async def cmd_probe(args) -> int:
    """Live read-only check of the real TableCheck endpoints for a venue."""
    from workflows.reservations.channels import TableCheckChannel
    ch = TableCheckChannel()
    info = await ch.venue_info(args.slug)
    print(json.dumps({k: v for k, v in info.items()
                      if k not in ("booking_policy", "cancel_policy")},
                     ensure_ascii=False, indent=1))
    state = await ch.fetch_month(args.slug, args.party, args.month)
    open_days = [d for d, i in state["dates"].items() if i["open"]]
    full_days = [d for d, i in state["dates"].items() if not i["open"]]
    print(f"{args.month} x{args.party}: open={open_days} full={full_days} "
          f"(closed/unreleased: {30 - len(state['dates'])}+ days)")
    if args.day:
        print(args.day, await ch.fetch_day_slots(args.slug, args.party, args.day))
    return 0


def main(argv: Optional[list] = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("text")
    a.add_argument("--user", default=_default_user())
    a.add_argument("--release", help="ISO datetime with offset when the calendar drops")
    a.add_argument("--burst-until", help="ISO datetime with offset; keep bursting until then")
    a.add_argument("--dry-run", action="store_true")
    l = sub.add_parser("list"); l.add_argument("--user")
    s = sub.add_parser("say"); s.add_argument("text"); s.add_argument("--user", default=_default_user())
    pr = sub.add_parser("probe"); pr.add_argument("slug"); pr.add_argument("party", type=int)
    pr.add_argument("month"); pr.add_argument("--day")
    args = p.parse_args(argv)
    if args.cmd == "add":
        return asyncio.run(cmd_add(args))
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "say":
        return asyncio.run(cmd_say(args))
    if args.cmd == "probe":
        return asyncio.run(cmd_probe(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())
