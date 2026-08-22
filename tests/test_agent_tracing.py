"""Optional LangSmith tracing: off by default, redacted when on, engine wires callbacks."""

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from agent.tracing import CONTEXT_CLOSE, CONTEXT_OPEN, build_tracer, scrub, trace_config
from config import AgentConfig
from tests.agent_fakes import ScriptedChatModel, make_engine


def test_tracer_is_none_when_flag_off_or_key_missing(monkeypatch):
    assert build_tracer(AgentConfig(tracing=False)) is None
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert build_tracer(AgentConfig(tracing=True)) is None


def test_tracer_built_with_project_and_sampling_rate(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_not_a_real_key")
    tracer = build_tracer(AgentConfig(tracing=True, tracing_project="friday-test",
                                      tracing_sampling_rate=0.5))
    assert tracer is not None
    assert tracer.project_name == "friday-test"
    assert tracer.client.tracing_sample_rate == 0.5
    # The redaction hooks are installed on the client that will upload.
    assert tracer.client._hide_inputs is scrub and tracer.client._hide_outputs is scrub


def test_scrub_redacts_pii_and_hides_context_block():
    run = {
        "inputs": {
            "messages": [
                {"type": "system", "content": f"You are Jarvis.\n{CONTEXT_OPEN}\ntea: Earl Grey, "
                                              f"phone +1 415 555 0123\n{CONTEXT_CLOSE}"},
                {"type": "human", "content": "email me at prem@example.com about 4111 1111 1111 1111"},
            ],
            "context_block": "tea: Earl Grey",
        },
        "nested": [{"note": "call +44 7700 900123"}],
    }
    out = scrub(run)
    system = out["inputs"]["messages"][0]["content"]
    assert "Earl Grey" not in system and "<context_block hidden" in system
    assert "You are Jarvis." in system
    human = out["inputs"]["messages"][1]["content"]
    assert "prem@example.com" not in human and "p***@***" in human
    assert "4111 1111 1111 1111" not in human and "CARD" in human
    assert "email me at" in human                                   # user text otherwise intact
    assert out["inputs"]["context_block"] == "<context_block hidden, 14 chars>"
    assert "900123" not in out["nested"][0]["note"] or "***0123" in out["nested"][0]["note"]
    # Original untouched.
    assert run["inputs"]["context_block"] == "tea: Earl Grey"


def test_trace_config_is_empty_without_a_tracer():
    assert trace_config(None, user_id="u", thread_id="t", run_kind="fresh") == {}


class RecordingHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.chain_starts = []
        self.metadata = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None,
                       metadata=None, **kwargs):
        self.chain_starts.append(kwargs.get("name") or (serialized or {}).get("name"))
        if metadata:
            self.metadata.append(dict(metadata))


@pytest.mark.asyncio
async def test_engine_invokes_graph_with_callbacks_when_tracer_present():
    handler = RecordingHandler()
    model = ScriptedChatModel().push(AIMessage(content="Traced, sir."))
    engine = make_engine(model=model, tracer=handler)
    assert await engine.handle("hello") == "Traced, sir."
    assert handler.chain_starts, "tracer never received a chain start"
    assert any(m.get("run_kind") == "fresh" and m.get("engine") == "langgraph"
               and "user_id_hash" in m and "default" not in m.values() for m in handler.metadata)


@pytest.mark.asyncio
async def test_engine_runs_without_callbacks_when_tracer_absent():
    model = ScriptedChatModel().push(AIMessage(content="Quiet, sir."))
    engine = make_engine(model=model, tracer=None)
    assert await engine.handle("hello") == "Quiet, sir."
