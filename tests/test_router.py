"""Tests for the intent router's response parsing."""

from llm.router import IntentRouter


def _router():
    # Bypass __init__ (which only sets up lazy LLM clients) — _parse is pure.
    return IntentRouter.__new__(IntentRouter)


def test_parse_plain_json():
    r = _router()
    res = r._parse('{"workflow":"reservations","entities":{"business_name":"Flores"},'
                   '"response":"On it."}')
    assert res.workflow_name == "reservations"
    assert res.entities == {"business_name": "Flores"}
    assert res.response == "On it."


def test_parse_fenced_json():
    """Regression: a clean ```json fenced reply used to parse as empty because the
    fence-stripping grabbed the text *after* the closing fence."""
    r = _router()
    raw = '```json\n{\n  "workflow": null,\n  "entities": {},\n  "response": "Hi there!"\n}\n```'
    res = r._parse(raw)
    assert res.workflow_name is None
    assert res.response == "Hi there!"


def test_parse_json_with_surrounding_prose():
    r = _router()
    raw = 'Sure — here you go: {"workflow":"time","entities":{},"response":"It is 5pm"} hope that helps'
    res = r._parse(raw)
    assert res.workflow_name == "time"
    assert res.response == "It is 5pm"


def test_parse_unparseable_is_safe():
    r = _router()
    res = r._parse("the model rambled without any json")
    assert res.workflow_name is None
    assert res.response == ""  # never speak raw text back
