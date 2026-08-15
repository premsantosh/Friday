"""
Tests for the standing-watch pure core (workflows/reservations/watcher.py, M8):
criteria, poll planning (fetch units), the snapshot differ, the matcher, and
the conversational spec parsing.
"""

from datetime import datetime

from core.harness import NormalizeCtx
from workflows.reservations.watcher import (
    WatchCriterion,
    dedupe_key,
    diff_states,
    expire_criteria,
    fetch_units,
    match_events,
    months_spanned,
    open_dates,
    open_times,
    parse_date_range,
    parse_times,
    relevant_open_dates,
)


def crit(**kw):
    base = dict(date_start="2026-09-05", date_end="2026-09-06",
                party_size=2, times=["19:00", "21:00"])
    base.update(kw)
    return WatchCriterion(**base)


def state(dates=None, venue_status="ok", month="2026-09"):
    return {"month": month, "dates": dates or {}, "venue_status": venue_status}


def day(times_avail=(), times_full=(), open_flag=None):
    slots = {t: "available" for t in times_avail}
    slots.update({t: "full" for t in times_full})
    return {"open": bool(times_avail) if open_flag is None else open_flag,
            "slots": slots}


# ------------------------------------------------------------------- criteria

def test_criterion_matches_range_and_times():
    c = crit()
    assert c.matches("2026-09-05", "19:00")
    assert c.matches("2026-09-06", "21:00")
    assert not c.matches("2026-09-07", "19:00")     # outside range
    assert not c.matches("2026-09-05", "17:00")     # wrong time


def test_criterion_any_time_and_day_level():
    assert crit(times=None).matches("2026-09-05", "17:00")
    # Day-level source (time unknown) matches even with specific times wanted.
    assert crit().matches("2026-09-05", None)


def test_inactive_criterion_never_matches():
    assert not crit(status="fulfilled").matches("2026-09-05", "19:00")


def test_expire_criteria_venue_local_today():
    c1, c2 = crit(), crit(date_start="2026-09-09", date_end="2026-09-09")
    assert expire_criteria([c1, c2], "2026-09-07") is True
    assert c1.status == "expired" and c2.status == "active"
    assert expire_criteria([c1, c2], "2026-09-07") is False   # idempotent


def test_criterion_roundtrip():
    c = crit(label="anniversary", priority=10)
    assert WatchCriterion.from_dict(c.to_dict()) == c


# -------------------------------------------------------------- poll planning

def test_months_spanned():
    assert months_spanned("2026-09-05", "2026-09-06") == ["2026-09"]
    assert months_spanned("2026-09-28", "2026-10-02") == ["2026-09", "2026-10"]
    assert months_spanned("2026-11-30", "2027-01-02") == ["2026-11", "2026-12", "2027-01"]


def test_fetch_units_dedup_and_party_separation():
    criteria = [
        crit(),                                                    # (2, 2026-09)
        crit(date_start="2026-09-09", date_end="2026-09-09"),      # (2, 2026-09) dup
        crit(date_start="2026-09-10", date_end="2026-09-10", party_size=4),
        crit(status="expired", party_size=6),                      # inactive → no unit
    ]
    assert fetch_units(criteria) == [(2, "2026-09"), (4, "2026-09")]


# --------------------------------------------------------------------- differ

def test_first_snapshot_is_baseline():
    assert diff_states(None, state({"2026-09-05": day(["19:00"])})) == []


def test_slot_opened_and_closed():
    prev = state({"2026-09-05": day(["19:00"]), "2026-09-06": day(["21:00"])})
    curr = state({"2026-09-05": day(["19:00", "21:00"]), "2026-09-06": day([], ["21:00"])})
    events = diff_states(prev, curr)
    assert {"type": "slot_opened", "date": "2026-09-05", "time": "21:00"} in events
    assert {"type": "slot_closed", "date": "2026-09-06", "time": "21:00"} in events


def test_calendar_published():
    prev = state({})
    curr = state({"2026-09-05": day(["19:00"]), "2026-09-06": day(["21:00"])})
    events = diff_states(prev, curr)
    published = [e for e in events if e["type"] == "calendar_published"]
    assert published and published[0]["dates"] == ["2026-09-05", "2026-09-06"]
    # The publish also surfaces the individual slots for matching.
    assert {"type": "slot_opened", "date": "2026-09-05", "time": "19:00"} in events


def test_venue_halted_and_resumed():
    ok, halted = state({"2026-09-05": day(["19:00"])}), state(venue_status="halted")
    assert diff_states(ok, halted) == [{"type": "venue_halted"}]
    events = diff_states(halted, ok)
    assert {"type": "venue_resumed"} in events


def test_day_level_open_close():
    prev = state({"2026-09-05": day(open_flag=False)})
    curr = state({"2026-09-05": day(open_flag=True)})
    # Both snapshots carry the date entry, so this is a slot change — a
    # published-but-full calendar freeing a day is not a "publish".
    assert diff_states(prev, curr) == [
        {"type": "slot_opened", "date": "2026-09-05", "time": None},
    ]
    assert {"type": "slot_closed", "date": "2026-09-05", "time": None} \
        in diff_states(curr, prev)


# -------------------------------------------------------------------- matcher

def test_match_events_multiple_criteria_one_unit():
    c_a = crit(label="a")                                          # 5–6 @19/21
    c_b = crit(date_start="2026-09-06", date_end="2026-09-06",
               times=None, label="b")                              # 6 @any
    events = [{"type": "slot_opened", "date": "2026-09-06", "time": "21:00"},
              {"type": "slot_opened", "date": "2026-09-06", "time": "17:00"},
              {"type": "slot_closed", "date": "2026-09-05", "time": "19:00"}]
    matched = match_events(events, [c_a, c_b])
    assert (events[0], 0) in matched and (events[0], 1) in matched
    assert (events[1], 1) in matched and (events[1], 0) not in matched
    assert all(ev["type"] == "slot_opened" for ev, _ in matched)


def test_relevant_open_dates_and_helpers():
    s = state({"2026-09-05": day(["19:00"]), "2026-09-20": day(["19:00"]),
               "2026-09-06": day([], ["21:00"])})
    assert open_dates(s) == ["2026-09-05", "2026-09-20"]
    assert open_times(s, "2026-09-05") == ["19:00"]
    assert open_times(s, "2026-09-06") == []
    assert relevant_open_dates(s, [crit()]) == ["2026-09-05"]


def test_dedupe_key_shapes():
    assert dedupe_key(2, "2026-09-05", "19:00") == "2|2026-09-05|19:00"
    assert dedupe_key(4, "2026-09-05", None) == "4|2026-09-05|-"


# ------------------------------------------------------------- spec parsing

CTX = NormalizeCtx(now=datetime(2026, 8, 15, 12, 0))


def test_parse_date_range_single_and_range():
    assert parse_date_range("September 5", CTX) == ("2026-09-05", "2026-09-05")
    assert parse_date_range("September 5 to 6", CTX) == ("2026-09-05", "2026-09-06")
    assert parse_date_range("Sept 4 - 12", CTX) == ("2026-09-04", "2026-09-12")
    assert parse_date_range("9/5 through 9/7", CTX) == ("2026-09-05", "2026-09-07")


def test_parse_date_range_rejects_garbage_and_inverted():
    assert parse_date_range("whenever works", CTX) is None
    assert parse_date_range("September 6 to 5", CTX) is None
    assert parse_date_range("", CTX) is None


def test_parse_times_shared_meridiem():
    assert parse_times("at 7 or 9pm", CTX) == ["19:00", "21:00"]
    assert parse_times("at 7:00 PM", CTX) == ["19:00"]
    assert parse_times("19:00 or 21:00", CTX) == ["19:00", "21:00"]


def test_parse_times_any_and_no_times():
    assert parse_times("any time works", CTX) is None
    # Party sizes and ordinal dates never become times.
    assert parse_times("for 2 on September 5th or 6th", CTX) is None
    assert parse_times("7", CTX) is None
    assert parse_times("7", CTX, bare=True) == ["19:00"]
