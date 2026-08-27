"""SELF-KNOWLEDGE prompt block: present on both engines, live parts appended."""

from __future__ import annotations

from config.settings import PersonalityConfig
from llm.providers import generate_personality_prompt


def test_self_context_appears_in_personality_prompt():
    prompt = generate_personality_prompt(
        PersonalityConfig(self_context="- You are Friday."))
    assert "SELF-KNOWLEDGE:" in prompt
    assert "- You are Friday." in prompt


def test_empty_self_context_adds_no_section():
    prompt = generate_personality_prompt(PersonalityConfig(self_context=""))
    assert "SELF-KNOWLEDGE" not in prompt


def test_agent_system_prompt_carries_self_knowledge():
    from agent.nodes import build_system_prompt

    personality = PersonalityConfig(self_context="- You are Friday.")
    prompt = build_system_prompt(personality, context_block="", has_tools=True)
    assert "SELF-KNOWLEDGE:" in prompt
    assert "self_status" in prompt  # TOOL_GUIDANCE bullet


def test_build_self_context_names_the_learning_loop(monkeypatch):
    from main import _build_self_context

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    block = _build_self_context()
    assert "com.friday.nightly" in block
    assert "self_status" in block
    assert "LoRA" in block


def test_runtime_lines_are_appended_and_idempotent():
    from core.assistant import VoiceAssistant

    class FakeAssistant:
        """Only what _append_runtime_self_context touches."""

        _RUNTIME_SELF_MARKER = VoiceAssistant._RUNTIME_SELF_MARKER
        _append_runtime_self_context = VoiceAssistant._append_runtime_self_context
        engine_label = "legacy router"

        def __init__(self, workflows):
            from types import SimpleNamespace

            self.workflows = workflows
            self.config = SimpleNamespace(
                personality=PersonalityConfig(self_context="- You are Friday."),
                llm=SimpleNamespace(anthropic_model="claude-test", ephemeral=False),
            )

    from workflows import SelfStatusWorkflow, WorkflowManager

    manager = WorkflowManager()
    manager.register(SelfStatusWorkflow(workflow_manager=manager))
    fake = FakeAssistant(manager)
    fake._append_runtime_self_context()
    ctx = fake.config.personality.self_context
    assert "legacy router" in ctx and "claude-test" in ctx
    assert "self_status" in ctx

    fake._append_runtime_self_context()  # reconstructing must not stack
    assert fake.config.personality.self_context.count("Right now you are running") == 1
