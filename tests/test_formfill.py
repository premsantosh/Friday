"""
Tests for the generic form filler and the durable user profile.

Everything here is hermetic: the browser tests load saved HTML through
`page.set_content` and never touch the network, and the mapper tests drive a
canned fake instead of an LLM.

`tests/fixtures/forms/backontrack_pleasanton.html` is a real capture of the
page that motivated this work (an Elementor form whose fields are labelled by
placeholder only, and whose own JavaScript rewrites the date/time inputs to
plain text). It's the regression test that matters most — synthetic fixtures
agree with the code that made them.
"""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

import pytest

from core.profile import PROFILE_KEYS, UnknownProfileKey, UserProfile, match_key
from workflows.reservations.channels.generic_web import is_request_form
from workflows.reservations.formfill.agent import (
    FormAgent,
    FormPlan,
    PlanEntry,
    _drift,
    _PageState,
    _request_for_llm,
)
from workflows.reservations.formfill.harvest import (
    FormField,
    FormOption,
    FormSnapshot,
    harvest,
)
from workflows.reservations.formfill.mapper import map_fields

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "forms"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


@asynccontextmanager
async def page_with(html: str):
    """A real Chromium page holding the given markup. No network."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.set_content(html)
            yield page
        finally:
            await browser.close()


class FakeLLM:
    """Stands in for ReservationLLM: returns canned JSON, records the payload."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, system, user, max_tokens=700):
        self.calls.append({"system": system, "user": user})
        return self.response


def seeded_profile(**values) -> UserProfile:
    profile = UserProfile(":memory:")
    for key, value in values.items():
        profile.set(key, value)
    return profile


DEFAULT_FACTS = {
    "full_name": "Jordan Avery",
    "email": "jordan@example.com",
    "phone": "9255550142",
}


# ============================================================ durable profile

def test_descriptors_carry_no_values():
    """The one invariant that keeps PII off the wire."""
    profile = seeded_profile(**DEFAULT_FACTS, date_of_birth="1990-04-12",
                             insurance_member_id="XQ99201")
    blob = repr(profile.descriptors())
    for value in ("Jordan Avery", "jordan@example.com", "9255550142",
                  "1990-04-12", "XQ99201"):
        assert value not in blob
    assert {d["key"] for d in profile.descriptors()} >= {"full_name", "email", "phone"}


def test_descriptors_only_list_facts_we_hold():
    profile = seeded_profile(email="jordan@example.com")
    keys = {d["key"] for d in profile.descriptors()}
    assert "email" in keys
    assert "insurance_member_id" not in keys


def test_first_and_last_name_are_derived_not_duplicated():
    profile = seeded_profile(full_name="Jordan Avery")
    assert profile.get("first_name") == "Jordan"
    assert profile.get("last_name") == "Avery"
    assert profile.get("first_name") != profile.get("last_name")


def test_single_token_name_yields_no_surname():
    """Better an empty box than an invented surname on a medical form."""
    profile = seeded_profile(full_name="Jordan")
    assert profile.get("first_name") == "Jordan"
    assert profile.get("last_name") is None


def test_explicit_names_win_over_derivation():
    profile = seeded_profile(full_name="Jordan Avery", last_name="Avery-Brooks")
    assert profile.get("last_name") == "Avery-Brooks"


def test_unknown_key_is_refused():
    profile = UserProfile(":memory:")
    with pytest.raises(UnknownProfileKey):
        profile.set("social_security_number", "123-45-6789")


def test_env_seeding_never_clobbers_a_user_value(monkeypatch):
    profile = UserProfile(":memory:")
    profile.set("full_name", "Jordan Avery", source="user")
    monkeypatch.setenv("RESERVATION_GUEST_NAME", "Stale Env Name")
    monkeypatch.setenv("RESERVATION_USER_EMAIL", "jordan@example.com")

    seeded = profile.seed_from_env()

    assert profile.get("full_name") == "Jordan Avery"
    assert profile.get("email") == "jordan@example.com"   # the previously dead env var
    assert "email" in seeded and "full_name" not in seeded


def test_env_seeding_refreshes_its_own_rows(monkeypatch):
    profile = UserProfile(":memory:")
    monkeypatch.setenv("RESERVATION_USER_PHONE", "9255550142")
    profile.seed_from_env()
    monkeypatch.setenv("RESERVATION_USER_PHONE", "9255559999")
    profile.seed_from_env()
    assert profile.get("phone") == "9255559999"


@pytest.mark.parametrize("label,expected", [
    ("First Name *", "first_name"),
    ("Last Name", "last_name"),
    ("Your Email", "email"),
    ("Phone Number", "phone"),
    ("Date of Birth", "date_of_birth"),
    ("Insurance Provider", "insurance_provider"),
    ("How did you hear about us?", "referral_source"),
    # The bug this cost us on the real page: "tel" lives inside "Tell".
    ("Tell Us About Your Condition", None),
    ("What brings you in?", None),
])
def test_label_matching_is_word_boundaried(label, expected):
    assert match_key(label) == expected


# ================================================================= personas

def two_persona_profile() -> UserProfile:
    """The real shape: shared facts at default, names/phones per persona."""
    profile = UserProfile(":memory:")
    profile.set("email", "jordan@example.com")                       # shared
    profile.set("full_name", "Jordan Avery Whitfield", context="formal")
    profile.set("first_name", "Jordan Avery", context="formal")
    profile.set("last_name", "Whitfield", context="formal")
    profile.set("phone", "214-555-0101", context="formal")
    profile.set("full_name", "Jordan Avery", context="casual")
    profile.set("first_name", "Jordan", context="casual")
    profile.set("last_name", "Avery", context="casual")
    profile.set("phone", "650-555-0102", context="casual")
    return profile


def test_a_persona_overlays_the_shared_defaults():
    profile = two_persona_profile()

    assert profile.get("full_name", "formal") == "Jordan Avery Whitfield"
    assert profile.get("full_name", "casual") == "Jordan Avery"
    # Not overridden anywhere → the shared value, stored once.
    assert profile.get("email", "formal") == profile.get("email", "casual")


def test_personas_do_not_leak_into_each_other():
    profile = two_persona_profile()
    formal, casual = profile.all("formal"), profile.all("casual")

    assert formal["last_name"] == "Whitfield"
    assert casual["last_name"] == "Avery"
    assert formal["phone"] != casual["phone"]


def test_an_unknown_persona_is_refused():
    from core.profile import UnknownProfileContext

    with pytest.raises(UnknownProfileContext):
        UserProfile(":memory:").set("full_name", "X", context="formaal")


def test_descriptors_still_carry_no_values_per_persona():
    profile = two_persona_profile()
    for context in ("formal", "casual"):
        blob = repr(profile.descriptors(context))
        assert "Whitfield" not in blob
        assert "jordan@example.com" not in blob
        assert "214-555-0101" not in blob and "650-555-0102" not in blob


def test_a_v1_profile_table_migrates_to_personas(tmp_path):
    """Rows written before personas existed survive as the shared default."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("""CREATE TABLE profile (
        key TEXT PRIMARY KEY, value TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'user', updated_at TEXT)""")
    old.execute("INSERT INTO profile (key, value, source) VALUES ('email','a@b.c','user')")
    old.commit()
    old.close()

    profile = UserProfile(str(path))

    assert profile.get("email") == "a@b.c"
    assert profile.get("email", "formal") == "a@b.c"
    profile.set("full_name", "Formal Name", context="formal")
    assert profile.get("full_name", "casual") is None


@pytest.mark.parametrize("text,expected", [
    ("Back on Track Physical Therapy consultation", "formal"),
    ("Pleasanton Clinic", "formal"),
    ("dentist appointment", "formal"),
    ("annual physical with Dr. Rao", "formal"),
    ("law firm consultation", "formal"),
    ("Lazy Bear dinner reservation", "casual"),
    ("Fellow Barber haircut", "casual"),
    ("Spa Radiance massage", "casual"),
    ("", "casual"),
])
def test_formality_is_chosen_from_the_business(text, expected):
    from core.profile import choose_context

    assert choose_context(text) == expected


def test_a_form_asking_for_insurance_is_formal_whatever_it_is_called():
    """Read off the form itself — stronger than guessing from a business name."""
    from core.profile import choose_context

    assert choose_context("Bay Wellness Studio", {"full_name", "email"}) == "casual"
    assert choose_context("Bay Wellness Studio",
                          {"full_name", "insurance_provider"}) == "formal"
    assert choose_context("Bay Wellness Studio",
                          {"full_name", "date_of_birth"}) == "formal"


def test_env_seeding_stops_once_a_persona_is_asserted(monkeypatch):
    """A stale RESERVATION_GUEST_NAME must not reinstate itself next to the
    personas that replaced it."""
    profile = UserProfile(":memory:")
    profile.set("full_name", "Jordan Avery", context="casual")
    monkeypatch.setenv("RESERVATION_GUEST_NAME", "Jordan")

    profile.seed_from_env()

    assert profile.get("full_name") is None            # nothing at default
    assert profile.get("full_name", "casual") == "Jordan Avery"


# =================================================================== harvest

@pytest.mark.asyncio
async def test_harvest_reads_a_clinic_intake_form():
    async with page_with(fixture("clinic_intake.html")) as page:
        snap = await harvest(page)

    assert snap.ok
    by_ref = {f.ref: f for f in snap.fields}

    # Labels resolved four different ways: <label for>, a wrapping label,
    # aria-labelledby, and aria-label.
    assert by_ref["first_name"].label == "First Name *"
    assert by_ref["email"].label == "Email Address *"
    assert by_ref["dob"].label == "Date of Birth"
    assert by_ref["insurance"].label == "Insurance Provider"

    assert by_ref["email"].required and not by_ref["newsletter"].required
    assert by_ref["reason"].type == "textarea"
    assert snap.submit_labels == ["Request Appointment"]


@pytest.mark.asyncio
async def test_harvest_skips_honeypots_and_hidden_fields():
    async with page_with(fixture("clinic_intake.html")) as page:
        snap = await harvest(page)

    refs = {f.ref for f in snap.fields}
    assert "website" not in refs       # off-screen honeypot
    assert "csrf_token" not in refs    # type=hidden


@pytest.mark.asyncio
async def test_harvest_groups_radios_and_reads_select_options():
    async with page_with(fixture("clinic_intake.html")) as page:
        snap = await harvest(page)

    radio = snap.by_ref("returning")
    assert radio.type == "radio"
    assert radio.label == "Have you been seen at this clinic before?"
    assert [o.label for o in radio.options] == ["Yes", "No"]

    select = snap.by_ref("location")
    assert [o.value for o in select.options] == ["", "pleasanton", "dublin", "livermore"]


@pytest.mark.asyncio
async def test_harvest_refuses_a_form_with_card_fields():
    """Card data has its own path (payment.py); it never comes through here."""
    async with page_with(fixture("card_form.html")) as page:
        snap = await harvest(page)

    assert snap.blocked_reason == "payment_card_field"
    assert not snap.ok


@pytest.mark.asyncio
async def test_harvest_prefers_the_real_form_over_a_search_box():
    async with page_with(fixture("search_and_contact.html")) as page:
        snap = await harvest(page)

    refs = {f.ref for f in snap.fields}
    assert refs == {"fullname", "email", "mobile", "notes"}
    assert "q" not in refs
    assert snap.form_count == 2


@pytest.mark.asyncio
async def test_harvest_handles_a_widget_with_no_form_element():
    async with page_with(fixture("no_form_element.html")) as page:
        snap = await harvest(page)

    assert snap.form_count == 0
    labels = {f.label for f in snap.fields}
    assert {"Your name", "Email", "Service"} <= labels


@pytest.mark.asyncio
async def test_harvest_reads_the_real_backontrack_page():
    """The page this whole feature exists for, captured verbatim."""
    async with page_with(fixture("backontrack_pleasanton.html")) as page:
        snap = await harvest(page)

    assert snap.ok and snap.blocked_reason is None
    by_ref = {f.ref: f for f in snap.fields}
    assert set(by_ref) == {
        "form_fields[name]", "form_fields[email]", "form_fields[phone]",
        "form_fields[callback_day]", "form_fields[callback_time]",
        "form_fields[message]",
    }
    # Elementor labels by placeholder alone — there is no <label> anywhere.
    assert by_ref["form_fields[name]"].label == "Your Name"
    assert by_ref["form_fields[message]"].label == "Tell Us About Your Condition"
    assert by_ref["form_fields[email]"].required


# ==================================================================== mapper

def snapshot_of(*fields, url="https://clinic.example/contact", heading="Book") -> FormSnapshot:
    return FormSnapshot(url=url, heading=heading, fields=list(fields),
                        submit_labels=["Submit"])


def text_field(ref, label, required=False, type_="text") -> FormField:
    return FormField(ref=ref, tag="input", type=type_, label=label,
                     name=ref, required=required)


def select_field(ref, label, options) -> FormField:
    return FormField(ref=ref, tag="select", type="select", label=label, name=ref,
                     options=[FormOption(label=o, value=o.lower()) for o in options])


def test_mapper_sends_keys_but_never_values():
    profile = seeded_profile(**DEFAULT_FACTS)
    snap = snapshot_of(text_field("email", "Email"))
    llm = FakeLLM({"mappings": [{"ref": "email", "source": "fact", "fact_key": "email"}]})

    map_fields(llm, snap, profile.descriptors(), {"business_name": "Clinic"})

    sent = llm.calls[0]["user"]
    assert "email" in sent                      # the key
    assert "jordan@example.com" not in sent       # never the value
    assert "Jordan Avery" not in sent


def test_mapper_drops_refs_the_page_never_had():
    snap = snapshot_of(text_field("email", "Email"))
    llm = FakeLLM({"mappings": [
        {"ref": "email", "source": "fact", "fact_key": "email"},
        {"ref": "ssn", "source": "literal", "literal": "123-45-6789"},
    ]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert [a.ref for a in result.assignments] == ["email"]


def test_mapper_drops_fact_keys_outside_the_registry():
    """A model can only name facts from the closed registry — it can't invent
    a source. The label here matches nothing, so the deterministic fallback
    has nothing to say either and the field is simply left alone."""
    assert "mothers_maiden_name" not in PROFILE_KEYS
    snap = snapshot_of(text_field("x", "Favourite colour"))
    llm = FakeLLM({"mappings": [
        {"ref": "x", "source": "fact", "fact_key": "mothers_maiden_name"}]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert result.assignments == []


def test_mapper_reports_a_fact_we_dont_hold_as_missing():
    snap = snapshot_of(text_field("dob", "Date of Birth", required=True))
    llm = FakeLLM({"mappings": [
        {"ref": "dob", "source": "fact", "fact_key": "date_of_birth"}]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert result.assignments == []
    assert [(m.ref, m.suggested_key) for m in result.missing] == [("dob", "date_of_birth")]


def test_mapper_refuses_an_invented_identity():
    """A model-authored literal in a name box is a fabricated person."""
    snap = snapshot_of(text_field("who", "Full Name", required=True))
    llm = FakeLLM({"mappings": [
        {"ref": "who", "source": "literal", "literal": "John Smith"}]})

    held = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})
    assert [(a.source, a.fact_key) for a in held.assignments] == [("fact", "full_name")]

    empty = map_fields(llm, snap, seeded_profile().descriptors(), {})
    assert empty.assignments == []
    assert [m.suggested_key for m in empty.missing] == ["full_name"]


def test_mapper_drops_an_off_menu_choice():
    snap = snapshot_of(select_field("loc", "Location", ["Pleasanton", "Dublin"]))
    llm = FakeLLM({"mappings": [
        {"ref": "loc", "source": "literal", "literal": "Timbuktu"}]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert result.assignments == []


def test_mapper_resolves_a_valid_choice_to_its_option_value():
    snap = snapshot_of(select_field("loc", "Location", ["Pleasanton", "Dublin"]))
    llm = FakeLLM({"mappings": [
        {"ref": "loc", "source": "literal", "literal": "Pleasanton"}]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert result.assignments[0].option_value == "pleasanton"


def test_an_optional_field_we_have_vocabulary_for_is_still_asked_about():
    """A clinic that prints an optional insurance box will chase you for it.
    Leaving it silently blank is worse than one question, and the answer is
    stored once and reused."""
    snap = snapshot_of(text_field("email", "Email", required=True),
                       text_field("ins", "Insurance Provider"),        # optional
                       text_field("dob", "Date of Birth"))             # optional
    llm = FakeLLM({"mappings": [
        {"ref": "email", "source": "fact", "fact_key": "email"},
        {"ref": "ins", "source": "skip"},
        {"ref": "dob", "source": "skip"},
    ]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert {m.suggested_key for m in result.missing} == {
        "insurance_provider", "date_of_birth"}


def test_an_optional_field_we_already_hold_is_not_asked_about():
    snap = snapshot_of(text_field("dob", "Date of Birth"))
    profile = seeded_profile(**DEFAULT_FACTS, date_of_birth="1990-04-12")
    llm = FakeLLM({"mappings": [{"ref": "dob", "source": "skip"}]})
    result = map_fields(llm, snap, profile.descriptors(), {})

    assert result.missing == []


def test_an_optional_field_we_have_no_vocabulary_for_is_left_alone():
    """No endless interrogation: a field outside the registry isn't a question."""
    snap = snapshot_of(text_field("colour", "Favourite colour"))
    llm = FakeLLM({"mappings": [{"ref": "colour", "source": "skip"}]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert result.missing == [] and result.assignments == []


def test_mapper_flags_unaccounted_required_fields():
    """Whatever the model says, a required field nobody filled is missing."""
    snap = snapshot_of(text_field("email", "Email", required=True),
                       text_field("dob", "Date of Birth", required=True))
    llm = FakeLLM({"mappings": [
        {"ref": "email", "source": "fact", "fact_key": "email"}]})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert [m.ref for m in result.missing] == ["dob"]


def test_mapper_falls_back_to_labels_without_an_llm():
    snap = snapshot_of(text_field("email", "Email Address"),
                       text_field("phone", "Phone Number"),
                       text_field("notes", "Tell us about your condition"))
    result = map_fields(None, snap, seeded_profile(**DEFAULT_FACTS).descriptors(),
                        {"special_requests": "Sore knee"})

    assert result.provenance == "fallback"
    assigned = {a.ref: (a.fact_key or a.literal) for a in result.assignments}
    assert assigned == {"email": "email", "phone": "phone", "notes": "Sore knee"}


def test_mapper_falls_back_when_the_llm_returns_nonsense():
    snap = snapshot_of(text_field("email", "Email Address"))
    llm = FakeLLM({"mappings": "not a list"})
    result = map_fields(llm, snap, seeded_profile(**DEFAULT_FACTS).descriptors(), {})

    assert result.provenance == "fallback"
    assert [a.fact_key for a in result.assignments] == ["email"]


def test_request_payload_excludes_identity():
    payload = _request_for_llm({
        "business_name": "Clinic", "service_type": "consult",
        "guest_name": "Jordan Avery", "phone": "9255550142",
        "email": "jordan@example.com", "raw_request": "book me in",
    })
    assert payload == {"business_name": "Clinic", "service_type": "consult"}


# ============================================================== plan + verify

@pytest.mark.asyncio
async def test_plan_substitutes_values_locally_and_fills_the_page():
    profile = seeded_profile(**DEFAULT_FACTS, date_of_birth="1990-04-12",
                             insurance_provider="Blue Shield",
                             referral_source="Google Search")
    agent = FormAgent(llm=None, profile=profile)

    async with page_with(fixture("clinic_intake.html")) as page:
        plan = await agent.plan(page, {"service_type": "initial consultation"})
        assert not plan.missing
        values = {e.ref: e.value for e in plan.entries}
        assert values["first_name"] == "Jordan"
        assert values["last_name"] == "Avery"
        assert values["consent"] == "true"

        snap = await harvest(page)
        filled, skipped, errors = await agent._apply(page, plan, snap)
        assert not skipped and not errors

        assert await page.locator("#fname").input_value() == "Jordan"
        assert await page.locator('[name="consent"]').is_checked()
        # Marketing opt-ins are never ticked on the user's behalf.
        assert not await page.locator('[name="newsletter"]').is_checked()
        # And the honeypot is never touched.
        assert await page.locator('[name="website"]').input_value() == ""


@pytest.mark.asyncio
async def test_a_clinic_form_is_filled_with_the_formal_persona():
    """The whole point: a medical intake form gets the full legal name."""
    profile = two_persona_profile()
    profile.set("date_of_birth", "1990-04-12")
    agent = FormAgent(llm=None, profile=profile)

    async with page_with(fixture("clinic_intake.html")) as page:
        plan = await agent.plan(page, {"business_name": "Pleasanton Clinic",
                                       "service_type": "physical therapy consultation"})

    assert plan.context == "formal"
    values = {e.ref: e.value for e in plan.entries}
    assert values["first_name"] == "Jordan Avery"
    assert values["last_name"] == "Whitfield"
    assert values["phone"] == "214-555-0101"


@pytest.mark.asyncio
async def test_a_salon_form_is_filled_with_the_everyday_persona():
    profile = two_persona_profile()
    agent = FormAgent(llm=None, profile=profile)

    async with page_with(fixture("no_form_element.html")) as page:
        plan = await agent.plan(page, {"business_name": "Fellow Barber",
                                       "service_type": "haircut"})

    assert plan.context == "casual"
    values = {e.label: e.value for e in plan.entries}
    assert values["Your name"] == "Jordan Avery"
    assert values["Mobile"] == "650-555-0102"


def test_the_page_itself_can_override_a_casual_looking_request():
    """The heading on that fixture says "Chiropractic" — that outranks whatever
    the request called the business."""
    from core.profile import choose_context

    assert choose_context("Fellow Barber haircut") == "casual"
    assert choose_context("Bay Area Chiropractic Fellow Barber haircut") == "formal"


@pytest.mark.asyncio
async def test_the_real_pt_page_gets_the_formal_persona():
    profile = two_persona_profile()
    agent = FormAgent(llm=None, profile=profile)

    async with page_with(fixture("backontrack_pleasanton.html")) as page:
        plan = await agent.plan(page, {
            "business_name": "Back on Track Physical Therapy",
            "service_type": "initial consultation"})

    assert plan.context == "formal"
    values = {e.ref: e.value for e in plan.entries}
    assert values["form_fields[name]"] == "Jordan Avery Whitfield"
    assert values["form_fields[phone]"] == "214-555-0101"


@pytest.mark.asyncio
async def test_plan_resolves_a_fact_onto_a_real_dropdown_option():
    profile = seeded_profile(**DEFAULT_FACTS, referral_source="Google Search")
    agent = FormAgent(llm=None, profile=profile)

    async with page_with(fixture("clinic_intake.html")) as page:
        plan = await agent.plan(page, {})
        entry = next(e for e in plan.entries if e.ref == "referral")
        assert entry.option_value == "google"


@pytest.mark.asyncio
async def test_plan_asks_for_a_required_fact_it_doesnt_hold():
    agent = FormAgent(llm=None, profile=seeded_profile(email="jordan@example.com"))

    async with page_with(fixture("clinic_intake.html")) as page:
        plan = await agent.plan(page, {})

    suggested = {m.suggested_key for m in plan.missing}
    assert "first_name" in suggested and "phone" in suggested


@pytest.mark.asyncio
async def test_plan_refuses_a_form_that_takes_card_details():
    agent = FormAgent(llm=None, profile=seeded_profile(**DEFAULT_FACTS))

    async with page_with(fixture("card_form.html")) as page:
        plan = await agent.plan(page, {})

    assert plan.blocked_reason == "payment_card_field"
    assert plan.entries == []


@pytest.mark.asyncio
async def test_plan_fills_the_real_backontrack_form():
    profile = seeded_profile(**DEFAULT_FACTS)
    agent = FormAgent(llm=None, profile=profile)
    request = {"service_type": "initial consultation", "date": "2026-08-05",
               "time": "10:00", "special_requests": "Lower back pain from running."}

    async with page_with(fixture("backontrack_pleasanton.html")) as page:
        plan = await agent.plan(page, request)
        assert not plan.missing
        values = {e.ref: e.value for e in plan.entries}
        assert values["form_fields[name]"] == "Jordan Avery"
        assert values["form_fields[email]"] == "jordan@example.com"
        # The regression: "tel" inside "Tell Us..." used to put a phone number here.
        assert values["form_fields[message]"] == "Lower back pain from running."

        filled, skipped, errors = await agent._apply(page, plan, await harvest(page))
        assert len(filled) == 6 and not skipped and not errors


def plan_for(*refs, required=()) -> FormPlan:
    return FormPlan(
        url="https://clinic.example/contact",
        entries=[PlanEntry(ref=r, label=r, field_type="text", value="x", origin="request")
                 for r in refs],
        known_refs=list(refs), required_refs=list(required))


def test_drift_detects_a_field_that_vanished():
    plan = plan_for("email", "phone")
    current = snapshot_of(text_field("email", "Email"))
    assert "phone" in (_drift(plan, current) or "")


def test_drift_detects_a_newly_required_field():
    plan = plan_for("email")
    current = snapshot_of(text_field("email", "Email"),
                          text_field("dob", "Date of Birth", required=True))
    assert "dob" in (_drift(plan, current) or "")


def test_drift_allows_a_new_optional_field():
    plan = plan_for("email")
    current = snapshot_of(text_field("email", "Email"), text_field("fax", "Fax"))
    assert _drift(plan, current) is None


def test_drift_refuses_a_form_that_grew_a_card_field():
    plan = plan_for("email")
    current = snapshot_of(text_field("email", "Email"))
    current.blocked_reason = "payment_card_field"
    assert "card" in (_drift(plan, current) or "")


@pytest.mark.asyncio
async def test_verification_needs_two_signals_to_claim_success():
    agent = FormAgent(llm=None, profile=seeded_profile())
    before = _PageState(url="about:blank", refs={"email"})

    async with page_with("<h1>Thank you</h1><p>We have received your request.</p>") as page:
        outcome = await agent._verify(page, before)

    # Form gone + success text.
    assert outcome.status == "submitted"
    assert set(outcome.signals) == {"success_text", "form_gone"}


@pytest.mark.asyncio
async def test_a_vanished_form_alone_is_not_success():
    agent = FormAgent(llm=None, profile=seeded_profile())
    before = _PageState(url="about:blank", refs={"email"})

    async with page_with("<h1>Bay Area Chiropractic</h1><p>Our clinics.</p>") as page:
        outcome = await agent._verify(page, before)

    assert outcome.status == "unconfirmed"
    assert outcome.signals == ["form_gone"]


@pytest.mark.asyncio
async def test_validation_errors_are_reported_not_swallowed():
    agent = FormAgent(llm=None, profile=seeded_profile())
    before = _PageState(url="about:blank", refs={"email"})
    html = """<form>
        <label for="e">Email</label><input id="e" name="email" required>
        <p class="err">This field is required</p>
      </form>"""

    async with page_with(html) as page:
        outcome = await agent._verify(page, before)

    assert outcome.status == "validation_failed"
    assert "required" in outcome.message.lower()


# =================================================================== channel

def test_a_contact_form_is_treated_as_a_request_not_a_booking():
    assert is_request_form(FormPlan(heading="Request a Consultation",
                                    submit_labels=["Request Appointment"]))
    assert is_request_form(FormPlan(url="https://clinic.example/contact/"))
    assert not is_request_form(FormPlan(heading="Reserve your table",
                                        submit_labels=["Book now"],
                                        url="https://bistro.example/reserve"))


def test_an_unreadable_form_is_assumed_to_be_a_request():
    assert is_request_form(None)


# ================================================================== workflow

def test_appointment_requests_dont_ask_for_a_party_size():
    from workflows.reservations.models import BookingKind, required_slots

    assert "party_size" in required_slots(BookingKind.DINING)
    assert "party_size" not in required_slots(BookingKind.APPOINTMENT)
    assert "service_type" in required_slots(BookingKind.APPOINTMENT)
    assert required_slots(BookingKind.INQUIRY) == ("business_name", "service_type")


@pytest.mark.parametrize("text,expected", [
    ("book me a table for 2 at Lazy Bear friday at 7pm", "dining"),
    ("make a dinner reservation at Flores", "dining"),
    ("book a consultation at Back on Track Physical Therapy", "appointment"),
    ("get me a haircut appointment at Fellow Barber saturday", "appointment"),
    ("book a massage at Spa Radiance", "appointment"),
    # Nothing to go on → the historical assumption, unchanged.
    ("make a reservation at Bungalow", "dining"),
])
def test_booking_kind_inference(text, expected):
    from workflows.reservations.workflow import ReservationWorkflow

    kind = ReservationWorkflow._kind_of({"raw_request": text})
    assert kind.value == expected


def test_a_pasted_url_is_extracted_and_kept_out_of_the_business_name():
    from workflows.reservations.workflow import ReservationWorkflow

    wf = ReservationWorkflow(discovery=object(), router=object(), llm=None,
                             payment=None, calendar=None, notifier=None,
                             sandbox=None, profile=None)
    slots = wf._extract_slots_regex(
        "book a consultation at Back on Track "
        "https://backontrack-pt.com/contact/pleasanton-clinic/")

    assert slots["target_url"] == "https://backontrack-pt.com/contact/pleasanton-clinic/"
    assert "http" not in (slots.get("business_name") or "")


def test_a_user_supplied_url_short_circuits_discovery():
    """No Places/Tavily round trip when the user already gave us the page."""
    from workflows.reservations.discovery import BusinessDiscovery
    from workflows.reservations.models import ReservationMethod

    class ExplodingSearch:
        def search(self, *a, **k):
            raise AssertionError("discovery should not search when given a URL")

    decision = BusinessDiscovery(search_provider=ExplodingSearch()).discover(
        "Back on Track Physical Therapy", "Pleasanton",
        target_url="https://backontrack-pt.com/contact/pleasanton-clinic/",
        kind="appointment")

    assert decision.method == ReservationMethod.GENERIC_WEB
    assert decision.url == "https://backontrack-pt.com/contact/pleasanton-clinic/"
    assert decision.source == "user_url"


def test_a_submitted_request_is_not_reported_as_a_confirmed_booking():
    from workflows.reservations.workflow import ReservationWorkflow

    facts = {"business_name": "Back on Track", "service_type": "consultation",
             "date": "2026-08-05"}
    pending = ReservationWorkflow._notify_text(
        facts, {"booking_kind": "appointment"}, pending=True)
    booked = ReservationWorkflow._notify_text(
        facts, {"booking_kind": "appointment"}, pending=False)

    assert "Request sent" in pending and "Awaiting their reply" in pending
    assert "confirmed" not in pending.lower()
    assert "Booked" in booked


def test_the_gate_discloses_every_field_and_value():
    from workflows.reservations.channels import CommitPlan
    from workflows.reservations.workflow import ReservationWorkflow

    plan = CommitPlan(
        channel="generic_web", summary="Submit a request for a consultation",
        details={"business_name": "Back on Track", "form_plan": FormPlan(
            entries=[
                PlanEntry(ref="name", label="Your Name", field_type="text",
                          value="Jordan Avery", origin="profile:full_name"),
                PlanEntry(ref="msg", label="Tell Us About Your Condition",
                          field_type="textarea", value="Lower back pain",
                          origin="request"),
            ]).to_dict()})

    message = ReservationWorkflow._confirm_message(plan, availability=None)

    assert "Your Name" in message and "Jordan Avery" in message
    assert "Lower back pain" in message
    assert "Shall I submit it" in message


def test_form_values_are_purged_when_the_session_ends():
    """The field structure is worth keeping; the data in it is not — that lives
    in the profile store."""
    from core.conversation.session import Session
    from workflows.reservations.workflow import ReservationWorkflow

    wf = ReservationWorkflow(discovery=object(), router=object(), llm=None,
                             payment=None, calendar=None, notifier=None,
                             sandbox=None, profile=None)
    session = Session(session_id="s1", user_id="u1", workflow_name="reservations")
    session.slots["email"] = "jordan@example.com"
    session.slots["commit_plan"] = {
        "channel": "generic_web", "summary": "s",
        "details": {"form_plan": FormPlan(entries=[
            PlanEntry(ref="dob", label="Date of Birth", field_type="text",
                      value="1990-04-12", origin="profile:date_of_birth")]).to_dict()}}

    wf.on_terminal(session)

    entry = session.slots["commit_plan"]["details"]["form_plan"]["entries"][0]
    assert entry["value"] == "[purged]"
    assert entry["label"] == "Date of Birth"      # structure survives
    assert session.slots["email"] is None


# ======================================================== end-to-end dialogue

class FakeFormChannel:
    """A generic-web channel whose form needs a date of birth we don't hold yet.

    Mirrors GenericWebChannel's contract without a browser: prepare() reports
    what it would fill and what it's missing, commit() submits a *request*.
    """
    method = None
    can_commit = True

    def __init__(self, profile):
        from workflows.reservations.models import ReservationMethod
        self.method = ReservationMethod.GENERIC_WEB
        self.profile = profile
        self.committed_with = None

    async def check_availability(self, slots):
        from workflows.reservations.channels import Availability, AvailabilityStatus
        return Availability(status=AvailabilityStatus.UNKNOWN)

    async def prepare(self, slots, decision):
        from workflows.reservations.channels import CommitPlan
        from workflows.reservations.formfill import MissingFact

        entries = [PlanEntry(ref="name", label="Your Name", field_type="text",
                             value=self.profile.get("full_name") or "", origin="profile:full_name")]
        missing = []
        dob = self.profile.get("date_of_birth")
        if dob:
            entries.append(PlanEntry(ref="dob", label="Date of Birth", field_type="text",
                                     value=dob, origin="profile:date_of_birth"))
        else:
            missing.append(MissingFact(ref="dob", label="Date of Birth",
                                       suggested_key="date_of_birth"))

        form_plan = FormPlan(url="https://clinic.example/contact",
                             heading="Request a Consultation", entries=entries,
                             missing=missing, submit_labels=["Request Appointment"])
        return CommitPlan(channel="generic_web",
                          summary="Submit a request for a consultation at Back on Track",
                          details={"business_name": "Back on Track",
                                   "url": form_plan.url,
                                   "form_plan": form_plan.to_dict()})

    async def commit(self, plan, payment=None):
        from workflows.reservations.channels import BookingResult
        self.committed_with = plan
        return BookingResult(success=True, pending=True,
                             message="Request submitted to Back on Track, sir.")


class RecordingNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return True


class ExplodingCalendar:
    """A confirmed *time* is the only thing that belongs in a calendar."""
    def create_event(self, facts):
        raise AssertionError("a pending request must not create a calendar event")


@pytest.mark.asyncio
async def test_form_booking_asks_for_a_missing_fact_then_submits_a_request():
    from core.conversation import InMemorySessionStore, SessionManager, TurnControl
    from workflows.base import WorkflowManager
    from workflows.reservations import ChannelRouter, ReservationWorkflow
    from workflows.reservations.models import ChannelDecision, ReservationMethod

    from tests.test_reservations import StubDiscovery, make_test_gate

    profile = seeded_profile(**DEFAULT_FACTS)
    channel = FakeFormChannel(profile)
    notifier = RecordingNotifier()
    decision = ChannelDecision(method=ReservationMethod.GENERIC_WEB,
                               business_name="Back on Track",
                               url="https://clinic.example/contact")

    workflows = WorkflowManager()
    workflows.register(ReservationWorkflow(
        discovery=StubDiscovery(decision),
        router=ChannelRouter({ReservationMethod.GENERIC_WEB: channel}),
        llm=None, gate=make_test_gate(), profile=profile,
        calendar=ExplodingCalendar(), notifier=notifier, payment=None, sandbox=None))
    mgr = SessionManager(InMemorySessionStore(), workflows,
                         default_timeout_s=1800)
    wf = workflows.workflows["reservations"]

    # An appointment, not a table: no "for how many people?" anywhere.
    turn = await mgr.open(
        wf, "book a consultation at Back on Track Physical Therapy next friday",
        {}, "default")
    assert "how many" not in turn.message.lower()

    # The form wants a date of birth we don't hold — it asks rather than guesses.
    assert turn.control == TurnControl.CONTINUE
    assert "Date of Birth" in turn.message

    turn = await mgr.handle("default", "1990-04-12")
    assert profile.get("date_of_birth") == "1990-04-12"      # remembered durably

    # Now the gate, disclosing every field and value.
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "Your Name" in turn.message and "Jordan Avery" in turn.message
    assert "1990-04-12" in turn.message

    turn = await mgr.handle("default", "yes")
    assert channel.committed_with is not None
    assert turn.control == TurnControl.COMPLETE

    # A request, reported as a request. ExplodingCalendar proves no event.
    assert len(notifier.messages) == 1
    assert "Request sent" in notifier.messages[0]
    assert "confirmed" not in notifier.messages[0].lower()


@pytest.mark.asyncio
async def test_a_form_needing_something_we_have_no_vocabulary_for_hands_off():
    from core.conversation import InMemorySessionStore, SessionManager, TurnControl
    from workflows.base import WorkflowManager
    from workflows.reservations import ChannelRouter, ReservationWorkflow
    from workflows.reservations.channels import CommitPlan
    from workflows.reservations.formfill import MissingFact
    from workflows.reservations.models import ChannelDecision, ReservationMethod

    from tests.test_reservations import StubDiscovery, make_test_gate

    class UnfillableChannel(FakeFormChannel):
        async def prepare(self, slots, decision):
            form_plan = FormPlan(
                url="https://clinic.example/contact", heading="Request a Consultation",
                entries=[PlanEntry(ref="name", label="Your Name", field_type="text",
                                   value="Jordan Avery", origin="profile:full_name")],
                missing=[MissingFact(ref="ref_no", label="Referring physician NPI",
                                     suggested_key=None)])
            return CommitPlan(channel="generic_web", summary="s",
                              details={"business_name": "Back on Track",
                                       "url": form_plan.url,
                                       "form_plan": form_plan.to_dict()})

    profile = seeded_profile(**DEFAULT_FACTS)
    channel = UnfillableChannel(profile)
    decision = ChannelDecision(method=ReservationMethod.GENERIC_WEB,
                               business_name="Back on Track",
                               url="https://clinic.example/contact")

    workflows = WorkflowManager()
    workflows.register(ReservationWorkflow(
        discovery=StubDiscovery(decision),
        router=ChannelRouter({ReservationMethod.GENERIC_WEB: channel}),
        llm=None, gate=make_test_gate(), profile=profile,
        calendar=None, notifier=None, payment=None, sandbox=None))
    mgr = SessionManager(InMemorySessionStore(), workflows, default_timeout_s=1800)

    turn = await mgr.open(workflows.workflows["reservations"],
                          "book a consultation at Back on Track next friday", {}, "default")

    assert turn.control == TurnControl.COMPLETE
    assert "Referring physician NPI" in turn.message
    assert "clinic.example/contact" in turn.message   # the link to finish it
    assert channel.committed_with is None             # nothing submitted


# ====================================================== bot-gated submission

@pytest.mark.asyncio
async def test_recaptcha_is_detected_on_the_real_page():
    """The live finding, pinned: this clinic's form is gated by reCAPTCHA v3."""
    async with page_with(fixture("backontrack_pleasanton.html")) as page:
        snap = await harvest(page)

    assert snap.human_submit_required == "recaptcha"
    assert snap.ok                    # still fillable — just not submittable by us


@pytest.mark.asyncio
async def test_an_ungated_form_is_not_flagged():
    async with page_with(fixture("clinic_intake.html")) as page:
        snap = await harvest(page)

    assert snap.human_submit_required is None


@pytest.mark.asyncio
async def test_a_gated_form_is_filled_but_never_submitted():
    """We stop before clicking: the widget scores the whole session, so a
    programmatically-filled form is rejected however the click arrives.
    Burning a submission to rediscover that helps nobody."""
    clicked = []

    class WatchfulAgent(FormAgent):
        async def _submit(self, page, snapshot):
            clicked.append(True)
            return True

    agent = WatchfulAgent(llm=None, profile=seeded_profile(**DEFAULT_FACTS))
    async with page_with(fixture("backontrack_pleasanton.html")) as page:
        plan = await agent.plan(page, {"special_requests": "New patient"})
        assert plan.human_submit_required == "recaptcha"
        outcome = await agent.execute(page, plan, submit=True)

    assert outcome.status == "filled_only"
    assert outcome.filled and not clicked      # filled, never clicked
