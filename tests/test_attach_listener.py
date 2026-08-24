"""Cross-channel wiring in main._attach_listener.

The combined mode labels each channel for the research recorder and fans a
reply out to the other channel (speak a Telegram reply; Telegram-prompt a
voice reply). The mirror must never affect the reply returned to the channel.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from main import _attach_listener


class FakeAssistant:
    def __init__(self, reply="Very good, sir."):
        self.reply = reply
        self.calls = []

    async def process_input(self, text, user_id="default", channel=None):
        self.calls.append((text, user_id, channel))
        return self.reply


class FakeChannel:
    """Captures the handler like TelegramChannel/VoicePEChannel.start does."""

    def __init__(self):
        self.handler = None

    def start(self, handler):
        self.handler = handler


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_channel_label_reaches_process_input():
    a, ch = FakeAssistant(), FakeChannel()
    _attach_listener(a, ch, channel_label="voice_pe")
    reply = _run(ch.handler("good evening", "voice:bedroom"))
    assert reply == "Very good, sir."
    assert a.calls == [("good evening", "voice:bedroom", "voice_pe")]


def test_mirror_gets_text_reply_sender_and_cannot_change_the_reply():
    a, ch = FakeAssistant(reply="It is noon, sir."), FakeChannel()
    seen, done = [], threading.Event()

    def mirror(text, reply, sender):
        seen.append((text, reply, sender))
        done.set()

    _attach_listener(a, ch, channel_label="telegram", mirror=mirror)
    reply = _run(ch.handler("what time is it", "123"))
    assert reply == "It is noon, sir."
    assert done.wait(2.0)
    assert seen == [("what time is it", "It is noon, sir.", "123")]


def test_mirror_failure_is_contained():
    a, ch = FakeAssistant(), FakeChannel()
    fired = threading.Event()

    def mirror(text, reply, sender):
        fired.set()
        raise RuntimeError("speaker on fire")

    _attach_listener(a, ch, channel_label="telegram", mirror=mirror)
    assert _run(ch.handler("hello", "123")) == "Very good, sir."
    assert fired.wait(2.0)  # raised on the mirror thread, reply unaffected


def test_no_mirror_for_empty_reply():
    a, ch = FakeAssistant(reply=None), FakeChannel()
    called = threading.Event()
    _attach_listener(a, ch, channel_label="telegram",
                     mirror=lambda *args: called.set())
    assert _run(ch.handler("hello", "123")) is None
    assert not called.wait(0.3)
