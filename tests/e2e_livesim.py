"""
End-to-end live-sim exercise of Friday's TableCheck reservation workflow.

Run manually (not collected by pytest): python tests/e2e_livesim.py [scenario...]

Runs the REAL workflow through the REAL SessionManager with:
  - a local HTTP server faithfully simulating TableCheck's web_booking
    availability_calendar endpoint (real aiohttp fetch path, via the
    documented TABLECHECK_AVAILABILITY_URL override), and
  - a local HTTP server simulating the Telegram Bot API sendMessage
    endpoint (real TelegramNotifier + requests path).

Venue under test: TOKYO CONFIDENTIAL (tablecheck.com/en/shops/tokyoconfidentialbar)
— same reservation platform as Bar Benfiddich.
"""
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta

import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("RESERVATION_KILL_SWITCH", None)
os.environ["FRIDAY_PROFILE_DB"] = os.path.join(tempfile.mkdtemp(), "profile.db")
os.environ["RESERVATION_TIMEZONE"] = "America/Los_Angeles"

from aiohttp import web  # noqa: E402

VENUE_SLUG = "tokyoconfidentialbar"
VENUE_URL = f"https://www.tablecheck.com/en/shops/{VENUE_SLUG}/reserve"
VENUE_NAME = "TOKYO CONFIDENTIAL"
TG_TOKEN = "123456:TESTTOKEN"
TG_CHAT = "777001"

# Programmable per-scenario availability: callable(slug, party, start, end) -> (status, payload)
STATE = {"handler": None, "tc_requests": [], "tg_messages": []}


def serve_month(open_map):
    """open_map: {iso_date: [times]} → standard widget-shaped payload."""
    def handler(slug, party, start, end):
        month = start[:7]
        cal = {d: t for d, t in open_map.items() if d.startswith(month)}
        return 200, {"availability_calendar": cal,
                     "shop": {"slug": slug, "status": "open"}}
    return handler


async def tc_handler(request):
    slug = request.match_info["slug"]
    q = request.rel_url.query
    STATE["tc_requests"].append(str(request.rel_url))
    assert "num_people" in q and "start_date" in q and "end_date" in q, "missing widget params"
    status, payload = STATE["handler"](slug, int(q["num_people"]),
                                       q["start_date"], q["end_date"])
    return web.json_response(payload, status=status)


async def tg_handler(request):
    body = await request.json()
    STATE["tg_messages"].append(body)
    return web.json_response({"ok": True, "result": {"message_id": len(STATE["tg_messages"])}})


async def start_servers():
    app = web.Application()
    app.router.add_get("/api/web_booking/v1/shops/{slug}/availability_calendar", tc_handler)
    app.router.add_post(f"/bot{TG_TOKEN}/sendMessage", tg_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ""))


def build_env(port, notifier=True):
    os.environ["TABLECHECK_AVAILABILITY_URL"] = (
        f"http://127.0.0.1:{port}/api/web_booking/v1/shops/{{slug}}/availability_calendar"
        f"?num_people={{party_size}}&start_date={{start_date}}&end_date={{end_date}}")

    from core.conversation import InMemorySessionStore, SessionManager
    from core.harness import ActionGate, AuditLog
    from workflows.base import WorkflowManager
    from workflows.reservations import ChannelRouter, ReservationMethod, ReservationWorkflow
    from workflows.reservations.channels import TableCheckChannel
    from workflows.reservations.discovery import BusinessDiscovery
    from workflows.reservations.notify import TelegramNotifier
    from workflows.reservations.workflow import RESERVATION_MACHINE

    store = InMemorySessionStore()
    gate = ActionGate.with_defaults(kill_switch_env="RESERVATION_KILL_SWITCH",
                                    audit=AuditLog(":memory:"),
                                    gate_states=RESERVATION_MACHINE.gate_states)
    tg = TelegramNotifier(TG_TOKEN, TG_CHAT) if notifier else None
    if tg:
        tg.api = f"http://127.0.0.1:{port}/bot{TG_TOKEN}"   # same code path, local sink
    wf = ReservationWorkflow(
        discovery=BusinessDiscovery(search_provider=None, business_client=None, llm=None),
        router=ChannelRouter({ReservationMethod.TABLECHECK: TableCheckChannel()}),
        notifier=tg, llm=None, gate=gate, session_store=store)
    wf._tc_unit_spacing_s = 0
    wfm = WorkflowManager()
    wfm.register(wf)
    mgr = SessionManager(store, wfm, default_timeout_s=1800)
    return mgr, wf, store


async def tick(mgr, session):
    session.wake_at = 0
    mgr.store.save(session)
    await mgr.tick_waiting()
    return mgr.store.get(session.session_id)


def waiting(mgr):
    return mgr.store.list_waiting()[0]


# ----------------------------------------------------------------- scenarios

async def s1_book_available(port):
    print("\nS1: book → available → confirm → manual hand-off deep link")
    from core.conversation import TurnControl
    mgr, wf, store = build_env(port)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    STATE["handler"] = serve_month({tomorrow: ["19:00", "21:00"]})
    turn = await mgr.open(wf.workflows if hasattr(wf, "workflows") else wf,
                          f"Book me a table for 2 at {VENUE_NAME} tomorrow at 7pm — "
                          f"their page: {VENUE_URL}", {}, "u1")
    check("confirm gate reached", turn.control == TurnControl.AWAIT_CONFIRMATION, turn.message)
    check("resolved date/time in gate message",
          "7:00" in turn.message.lower() or "7 pm" in turn.message.lower()
          or "19:00" in turn.message, turn.message)
    turn = await mgr.handle("u1", "yes")
    check("manual handoff (no auto-book on TableCheck)", turn.control == TurnControl.COMPLETE,
          str(turn.control))
    link_ok = (f"tablecheck.com/en/{VENUE_SLUG}/reserve" in turn.message
               and f"start_date={tomorrow}" in turn.message
               and "start_time=19%3A00" in turn.message or "start_time=19:00" in turn.message)
    check("deep link carries date/time/party", link_ok and "num_people=2" in turn.message,
          turn.message)
    check("real widget HTTP request made", any("num_people=2" in u for u in STATE["tc_requests"]))


async def s2_wait_and_book(port):
    print("\nS2: unavailable → offer watch → slot opens → re-confirm → hand-off")
    from core.conversation import TurnControl
    mgr, wf, store = build_env(port)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    STATE["handler"] = serve_month({tomorrow: []})
    turn = await mgr.open(wf, f"Book me a table for 2 at {VENUE_NAME} tomorrow at 7pm — "
                              f"their page: {VENUE_URL}", {}, "u2")
    check("offers to keep watching", turn.control == TurnControl.AWAIT_CONFIRMATION
          and "keep checking" in turn.message, turn.message)
    turn = await mgr.handle("u2", "yes")
    check("watch armed (background)", turn.control == TurnControl.BACKGROUND, turn.message)
    session = waiting(mgr)
    STATE["handler"] = serve_month({tomorrow: ["19:00"]})
    session = await tick(mgr, session)
    check("re-confirm on opening", session.fsm_state == "confirm", session.fsm_state)
    turn = await mgr.handle("u2", "yes")
    check("hand-off after reopening", "reserve" in turn.message and VENUE_SLUG in turn.message,
          turn.message)


async def s3_standing_watch(port):
    print("\nS3: standing watch → closed → publish → telegram notifications (real HTTP)")
    from core.conversation import TurnControl
    mgr, wf, store = build_env(port)
    d = date.today() + timedelta(days=17)   # keep below autodetect threshold? watch path researches anyway
    iso = d.isoformat()
    STATE["handler"] = serve_month({})
    turn = await mgr.open(wf, f"Watch {VENUE_NAME} {VENUE_URL} for {d.month}/{d.day} "
                              f"at 7pm or 9pm for 2", {}, "u3")
    check("watch confirm gate", turn.control == TurnControl.AWAIT_CONFIRMATION, turn.message)
    turn = await mgr.handle("u3", "yes")
    check("watch running", turn.control == TurnControl.BACKGROUND, turn.message)
    session = waiting(mgr)
    check("slug resolved from pasted URL",
          session.slots["watchlist_state"]["slug"] == VENUE_SLUG,
          str(session.slots["watchlist_state"]))
    session = await tick(mgr, session)      # baseline; research finds nothing (offline)
    research = [m for m in STATE["tg_messages"] if "couldn't pin down" in m.get("text", "")]
    check("release research degrades honestly", len(research) == 1,
          json.dumps(STATE["tg_messages"]))
    STATE["handler"] = serve_month({iso: ["19:00", "21:00"]})
    session = await tick(mgr, session)
    live = [m for m in STATE["tg_messages"] if "calendar is live" in m.get("text", "")]
    check("publish summary sent over real HTTP", len(live) == 1
          and live[0].get("chat_id") == TG_CHAT, json.dumps(STATE["tg_messages"]))
    check("summary carries booking deep link",
          live and f"tablecheck.com/en/{VENUE_SLUG}/reserve" in live[0]["text"],
          live[0]["text"] if live else "none")


async def s4_snipe_rolling(port):
    print("\nS4: stated rolling release policy → scheduled snipe → drop → handoff notify")
    from core.conversation import TurnControl
    mgr, wf, store = build_env(port)
    d = date.today() + timedelta(days=20)
    STATE["handler"] = serve_month({})
    turn = await mgr.open(wf, f"Book me a table for 2 at {VENUE_NAME} on {d.month}/{d.day} "
                              f"at 7pm {VENUE_URL} — tables open 10 days in advance at 9am JST",
                          {}, "u4")
    check("snipe gate offered", turn.control == TurnControl.AWAIT_CONFIRMATION
          and "reservations open" in turn.message, turn.message)
    check("fire displayed in JST", "JST" in turn.message, turn.message)
    turn = await mgr.handle("u4", "yes")
    check("snipe scheduled", turn.control == TurnControl.BACKGROUND
          and "stand ready" in turn.message, turn.message)
    session = waiting(mgr)
    ss = session.slots["snipe_state"]
    # Fast-forward: pretend the window just opened.
    now = time.time()
    ss["fire_ts"], ss["deadline_ts"] = now - 1, now + 3
    STATE["handler"] = serve_month({d.isoformat(): ["19:00"]})
    session = await tick(mgr, session)
    tg = [m for m in STATE["tg_messages"] if "⏰" in m.get("text", "")]
    check("drop outcome reported on telegram", len(tg) >= 1, json.dumps(STATE["tg_messages"]))
    check("handoff deep link in outcome",
          any(VENUE_SLUG in m["text"] for m in tg), json.dumps(tg))


async def s5_batch_policy(port):
    print("\nS5: BATCH (monthly) release policy — the Benfiddich scenario")
    from core.conversation import TurnControl
    mgr, wf, store = build_env(port)
    STATE["handler"] = serve_month({})
    sept5 = date(date.today().year, 9, 5)
    drop_day = date(date.today().year, 8, 20)
    # 5a: snipe phrasing with an absolute drop date
    turn = await mgr.open(wf, f"Book me a table for 2 at {VENUE_NAME} on 9/5 at 7pm "
                              f"{VENUE_URL} — reservations for September open on August 20 "
                              f"at 10am JST", {}, "u5")
    is_snipe = (turn.control == TurnControl.AWAIT_CONFIRMATION
                and "stand ready" in turn.message and "JST" in turn.message)
    check("absolute-date policy → scheduled snipe at the drop", is_snipe, turn.message)
    if is_snipe:
        turn = await mgr.handle("u5", "yes")
        session = waiting(mgr)
        ss = session.slots["snipe_state"]
        from zoneinfo import ZoneInfo
        expect = datetime(drop_day.year, 8, 20, 10, 0, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
        check("fire ts == Aug 20 10:00 JST", abs(ss["fire_ts"] - expect) < 61, str(ss))
    else:
        await mgr.cancel("u5") if asyncio.iscoroutinefunction(mgr.cancel) else mgr.cancel("u5")

    # 5b: standing watch with a monthly batch phrasing
    STATE["tg_messages"].clear()
    turn = await mgr.open(wf, f"Watch {VENUE_NAME} {VENUE_URL} for September 5 at 7pm for 2 — "
                              f"they open reservations for the following month on the 20th "
                              f"at 10am JST", {}, "u5b")
    ok_gate = turn.control == TurnControl.AWAIT_CONFIRMATION
    check("watch gate reached (batch phrasing)", ok_gate, turn.message)
    if ok_gate:
        wants_seen = "Sep" in turn.message and "7:00" in turn.message.replace("19:00", "7:00")
        check("criteria not polluted by release clock time",
              "10:00" not in turn.message and "10 AM" not in turn.message, turn.message)
        turn = await mgr.handle("u5b", "yes")
        session = [s for s in mgr.store.list_waiting()
                   if s.slots.get("watchlist_state", {}).get("slug") == VENUE_SLUG][-1]
        rel = session.slots["watchlist_state"]["release"]
        from zoneinfo import ZoneInfo
        expect = datetime(date.today().year, 8, 20, 10, 0,
                          tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
        check("watch pre-armed with burst at the drop",
              rel.get("fire_ts") and abs(rel["fire_ts"] - expect) < 61, str(rel))


async def main():
    runner, port = await start_servers()
    only = sys.argv[1:] or None
    try:
        for fn in (s1_book_available, s2_wait_and_book, s3_standing_watch,
                   s4_snipe_rolling, s5_batch_policy):
            if only and fn.__name__ not in only:
                continue
            STATE["tc_requests"].clear()
            STATE["tg_messages"].clear()
            try:
                await fn(port)
            except Exception:
                check(f"{fn.__name__} crashed", False, traceback.format_exc())
    finally:
        await runner.cleanup()
    fails = [c for c in CHECKS if not c[1]]
    print(f"\n== {len(CHECKS) - len(fails)}/{len(CHECKS)} checks passed ==")
    for name, _, detail in fails:
        print(f"FAILED: {name}\n    {detail[:600]}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
