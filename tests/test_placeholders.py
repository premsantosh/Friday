"""Placeholder identity: tokens in prompts, values only at the delivery boundary.

UserProfile is isolated per-test by conftest's FRIDAY_PROFILE_DB fixture.
"""

from __future__ import annotations

import pytest

from core.placeholders import PlaceholderResolver, identity_block
from core.profile import FORMAL_CONTEXT, UserProfile
from memory.store import FridayStore
from tests.agent_fakes import FakeLLM, make_assistant


@pytest.fixture
def profile():
    p = UserProfile()
    p.set("full_name", "Alistair Percival Wentworth", context=FORMAL_CONTEXT)
    p.set("email", "alistair@example.com")
    return p


def test_identity_block_lists_tokens_but_never_values(profile):
    block = identity_block()
    assert "{{full_name}}" in block
    assert "{{first_name}}" in block            # derived from full_name
    assert "{{email}}" in block
    assert "Alistair" not in block
    assert "alistair@example.com" not in block


def test_identity_block_empty_without_profile():
    assert identity_block() == ""


def test_walled_off_fields_never_become_tokens(profile):
    profile.set("phone", "555-0100")
    profile.set("date_of_birth", "1987-01-01")
    resolver = PlaceholderResolver()
    assert "phone" not in resolver.tokens()
    assert "date_of_birth" not in resolver.tokens()
    assert "555-0100" not in identity_block()
    assert resolver.resolve("call {{phone}}") == "call"  # unknown token stripped


def test_resolve_substitutes_and_is_idempotent(profile):
    resolver = PlaceholderResolver()
    once = resolver.resolve("You are {{full_name}}, sir.")
    assert once == "You are Alistair Percival Wentworth, sir."
    assert resolver.resolve(once) == once


def test_spouse_token_comes_from_facts_with_confidence_gate(tmp_path, profile):
    store = FridayStore(db_path=str(tmp_path / "memory.db"))
    store.remember(key="wife_name", value="Beatrice", category="personal",
                   confidence=0.4)
    resolver = PlaceholderResolver(store=store)
    assert "spouse_name" not in resolver.tokens()       # below the 0.6 gate

    store.remember(key="wife_name", value="Beatrice", category="personal",
                   confidence=0.9)
    resolver = PlaceholderResolver(store=store)
    assert resolver.resolve("Your wife is {{spouse_name}}.") == "Your wife is Beatrice."


@pytest.mark.asyncio
async def test_process_input_resolves_at_the_delivery_boundary(profile):
    llm = FakeLLM(ephemeral=False)
    llm.generate_response = lambda text: "You are {{full_name}}, sir."
    assistant = make_assistant(llm=llm)
    reply = await assistant.process_input("what's my name?")
    assert reply == "You are Alistair Percival Wentworth, sir."


@pytest.mark.asyncio
async def test_process_input_passes_through_on_resolver_failure(profile, monkeypatch):
    import core.placeholders as mod

    def boom(*a, **kw):
        raise RuntimeError("resolver down")

    monkeypatch.setattr(mod.PlaceholderResolver, "resolve", boom)
    llm = FakeLLM(ephemeral=False)
    llm.generate_response = lambda text: "You are {{full_name}}, sir."
    assistant = make_assistant(llm=llm)
    reply = await assistant.process_input("what's my name?")
    assert reply == "You are {{full_name}}, sir."       # unchanged, no crash
