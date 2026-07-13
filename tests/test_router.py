"""Tests for the intent router's response parsing."""

from llm.router import IntentRouter


def _router():
    # Bypass __init__ (which only sets up lazy LLM clients) — _parse is pure.
    return IntentRouter.__new__(IntentRouter)


def test_parse_plain_json():
    r = _router()
    res = r._parse('{"workflow":"reservations","entities":{"business_name":"Flores"}}')
    assert res.workflow_name == "reservations"
    assert res.entities == {"business_name": "Flores"}


def test_parse_fenced_json():
    """Regression: a clean ```json fenced reply used to parse as empty because the
    fence-stripping grabbed the text *after* the closing fence."""
    r = _router()
    raw = '```json\n{\n  "workflow": null,\n  "entities": {}\n}\n```'
    res = r._parse(raw)
    assert res.workflow_name is None
    assert res.entities == {}


def test_parse_json_with_surrounding_prose():
    r = _router()
    raw = 'Sure — here you go: {"workflow":"time","entities":{}} hope that helps'
    res = r._parse(raw)
    assert res.workflow_name == "time"


def test_parse_ignores_legacy_response_field():
    """The router no longer drafts spoken replies; a model that still emits a
    "response" key must not break parsing (and the field is discarded)."""
    r = _router()
    res = r._parse('{"workflow":"time","entities":{},"response":"It is 5pm"}')
    assert res.workflow_name == "time"
    assert not hasattr(res, "response")


def test_parse_unparseable_is_safe():
    r = _router()
    res = r._parse("the model rambled without any json")
    assert res.workflow_name is None
    assert res.entities == {}
