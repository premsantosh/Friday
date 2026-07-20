"""
Tests for the Voice PE satellite channel (core/voice_pe_channel.py).

Hermetic: the aioesphomeapi client is injected via `client_factory` and the
pipeline callbacks (handle_start/handle_audio/handle_stop) are driven directly
on a local asyncio loop — no network, no audio hardware. The autouse fixture in
conftest.py strips VOICE_PE_* env so from_env() tests start clean.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

aioesphomeapi = pytest.importorskip("aioesphomeapi")
from aioesphomeapi import VoiceAssistantEventType as ET

from core.voice_pe_channel import VoicePEChannel, VoicePEDeviceConfig


# --------------------------------------------------------------------- fakes

class FakeWakeWord:
    def __init__(self, id, wake_word):
        self.id = id
        self.wake_word = wake_word


class FakeVoiceConfig:
    def __init__(self, available, active):
        self.available_wake_words = available
        self.active_wake_words = active


class FakeAPIClient:
    def __init__(self, host, port, noise_psk=None, timeline=None,
                 active_wake_words=("okay_nabu",)):
        self.host, self.port, self.noise_psk = host, port, noise_psk
        self.timeline = timeline if timeline is not None else []
        self.callbacks = {}
        self.set_config_calls = []
        self.connected = False

        self._active_wake_words = list(active_wake_words)

    async def connect(self, login=False, on_stop=None, log_errors=True):
        self.connected = True
        self.on_stop = on_stop

    async def disconnect(self, force=False):
        self.connected = False

    def subscribe_voice_assistant(self, *, handle_start, handle_stop,
                                  handle_audio=None, handle_announcement_finished=None):
        self.callbacks = {
            "start": handle_start, "stop": handle_stop, "audio": handle_audio,
        }
        return lambda: None

    def send_voice_assistant_event(self, event_type, data=None):
        self.timeline.append(("EVT", event_type, data))

    async def get_voice_assistant_configuration(self, timeout=None):
        return FakeVoiceConfig(
            [FakeWakeWord("okay_nabu", "Okay Nabu"), FakeWakeWord("hey_jarvis", "Hey Jarvis")],
            self._active_wake_words,
        )

    async def set_voice_assistant_configuration(self, active_wake_words):
        self.set_config_calls.append(list(active_wake_words))
        self._active_wake_words = list(active_wake_words)


class FakeSTT:
    def __init__(self, text="what time is it"):
        self.text = text
        self.calls = []

    def transcribe(self, audio_data, sample_rate):
        self.calls.append((audio_data, sample_rate))
        return self.text


# ------------------------------------------------------------------- helpers

def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _speech_chunk(n=1600):
    return (np.ones(n, dtype=np.int16) * 3000).tobytes()   # RMS ≈ 0.09 > 0.01


def _silence_chunk(n=1600):
    return np.zeros(n, dtype=np.int16).tobytes()


def make_channel(stt=None, has_active=None, handler_reply="It is noon, sir."):
    """Channel with one device, injected fakes, and fast VAD limits.

    Chunks are 1600 samples = 0.1 s, so: 1 silence chunk after speech ends the
    utterance, 3 all-silence chunks trip the no-speech timeout.
    """
    timeline = []

    def factory(host, port, psk):
        return FakeAPIClient(host, port, psk, timeline=timeline)

    ch = VoicePEChannel(
        [VoicePEDeviceConfig(name="kitchen", host="10.0.0.5")],
        silence_seconds=0.1, no_speech_seconds=0.3, max_utterance_seconds=1.0,
        client_factory=factory,
    )
    stt = stt if stt is not None else FakeSTT()

    def speak(text):
        timeline.append(("SPEAK", text, None))

    ch.bind(stt=stt, speak=speak, has_active=has_active)

    handled = []

    async def handler(text, sender):
        handled.append((text, sender))
        return handler_reply

    ch._handler = handler
    conn = ch._connections[0]
    conn._client = FakeAPIClient("10.0.0.5", 6053, None, timeline=timeline)
    return ch, conn, timeline, stt, handled


def _events(timeline):
    return [(t[1], t[2]) for t in timeline if t[0] == "EVT"]


async def _utterance(conn, chunks):
    port = await conn._handle_start("conv-1", 0, None, "hey jarvis")
    assert port == 0  # API-audio mode
    for c in chunks:
        await conn._handle_audio(c, None)
    if conn._finish_task is not None:
        await conn._finish_task


# ------------------------------------------------------------------- from_env

def test_from_env_returns_none_when_unconfigured():
    assert VoicePEChannel.from_env() is None


def test_from_env_parses_devices_and_psks(monkeypatch):
    monkeypatch.setenv("VOICE_PE_DEVICES", "kitchen=192.168.1.50, office=voice-pe.local:1234")
    monkeypatch.setenv("VOICE_PE_NOISE_PSK", "globalkey")
    monkeypatch.setenv("VOICE_PE_NOISE_PSK_KITCHEN", "kitchenkey")
    monkeypatch.setenv("VOICE_PE_SILENCE_SECONDS", "0.8")
    ch = VoicePEChannel.from_env()
    assert ch is not None
    assert [(d.name, d.host, d.port) for d in ch.devices] == [
        ("kitchen", "192.168.1.50", 6053),
        ("office", "voice-pe.local", 1234),
    ]
    assert ch.devices[0].noise_psk == "kitchenkey"
    assert ch.devices[1].noise_psk == "globalkey"
    assert ch.silence_seconds == 0.8
    assert ch.wake_word_id == "hey_jarvis"


def test_from_env_skips_malformed_entries(monkeypatch):
    monkeypatch.setenv("VOICE_PE_DEVICES", "nohost, =1.2.3.4, kitchen=10.0.0.5:notaport, office=10.0.0.6")
    ch = VoicePEChannel.from_env()
    assert ch is not None
    assert [d.name for d in ch.devices] == ["office"]


def test_from_env_none_when_all_entries_malformed(monkeypatch):
    monkeypatch.setenv("VOICE_PE_DEVICES", "garbage, also=")
    assert VoicePEChannel.from_env() is None


# ------------------------------------------------------------------- pipeline

def test_full_utterance_event_sequence_and_handler():
    ch, conn, timeline, stt, handled = make_channel()
    _run(_utterance(conn, [_speech_chunk(), _speech_chunk(), _silence_chunk()]))

    types = [e[0] for e in _events(timeline)]
    assert types == [
        ET.VOICE_ASSISTANT_RUN_START,
        ET.VOICE_ASSISTANT_STT_START,
        ET.VOICE_ASSISTANT_STT_END,
        ET.VOICE_ASSISTANT_INTENT_START,
        ET.VOICE_ASSISTANT_TTS_START,
        ET.VOICE_ASSISTANT_INTENT_END,
        ET.VOICE_ASSISTANT_RUN_END,
    ]
    assert ET.VOICE_ASSISTANT_TTS_END not in types  # nothing plays on the puck

    by_type = dict(_events(timeline))
    assert by_type[ET.VOICE_ASSISTANT_STT_END] == {"text": "what time is it"}
    assert by_type[ET.VOICE_ASSISTANT_TTS_START] == {"text": "It is noon, sir."}
    assert by_type[ET.VOICE_ASSISTANT_INTENT_END] == {
        "conversation_id": "conv-1", "continue_conversation": "0",
    }

    assert handled == [("what time is it", "voice:kitchen")]
    (audio, rate), = stt.calls
    assert rate == 16000
    assert audio.dtype == np.float32 and len(audio) == 3 * 1600

    # Reply is spoken on the Mac before INTENT_END re-opens the puck's mic.
    speak_i = next(i for i, t in enumerate(timeline) if t[0] == "SPEAK")
    intent_end_i = next(
        i for i, t in enumerate(timeline)
        if t[0] == "EVT" and t[1] == ET.VOICE_ASSISTANT_INTENT_END
    )
    assert speak_i < intent_end_i


def test_continue_conversation_follows_has_active():
    ch, conn, timeline, _, _ = make_channel(has_active=lambda uid: True)
    _run(_utterance(conn, [_speech_chunk(), _silence_chunk()]))
    by_type = dict(_events(timeline))
    assert by_type[ET.VOICE_ASSISTANT_INTENT_END]["continue_conversation"] == "1"


def test_no_sessions_never_continues():
    # bind(has_active=None) — sessions disabled — must degrade to "0".
    ch, conn, timeline, _, _ = make_channel(has_active=None)
    _run(_utterance(conn, [_speech_chunk(), _silence_chunk()]))
    by_type = dict(_events(timeline))
    assert by_type[ET.VOICE_ASSISTANT_INTENT_END]["continue_conversation"] == "0"


def test_no_speech_times_out_with_quiet_error():
    ch, conn, timeline, stt, handled = make_channel()
    _run(_utterance(conn, [_silence_chunk(), _silence_chunk(), _silence_chunk()]))
    events = _events(timeline)
    assert [e[0] for e in events] == [
        ET.VOICE_ASSISTANT_RUN_START,
        ET.VOICE_ASSISTANT_STT_START,
        ET.VOICE_ASSISTANT_ERROR,
        ET.VOICE_ASSISTANT_RUN_END,
    ]
    assert dict(events)[ET.VOICE_ASSISTANT_ERROR]["code"] == "stt-no-text-recognized"
    assert stt.calls == [] and handled == []


def test_empty_transcript_reports_no_text_recognized():
    ch, conn, timeline, _, handled = make_channel(stt=FakeSTT(text="  "))
    _run(_utterance(conn, [_speech_chunk(), _silence_chunk()]))
    by_type = dict(_events(timeline))
    assert by_type[ET.VOICE_ASSISTANT_ERROR]["code"] == "stt-no-text-recognized"
    assert handled == []


def test_abort_cancels_and_next_run_works():
    ch, conn, timeline, stt, handled = make_channel()

    async def scenario():
        await conn._handle_start("conv-1", 0, None, "hey jarvis")
        await conn._handle_audio(_speech_chunk(), None)
        await conn._handle_stop(True)          # device aborted mid-stream
        assert conn._finish_task is None and conn._chunks == []
        timeline.clear()
        await _utterance(conn, [_speech_chunk(), _silence_chunk()])

    _run(scenario())
    assert handled == [("what time is it", "voice:kitchen")]
    assert _events(timeline)[-1][0] == ET.VOICE_ASSISTANT_RUN_END


def test_device_stop_triggers_transcription_of_buffer():
    ch, conn, timeline, stt, handled = make_channel()

    async def scenario():
        await conn._handle_start("conv-1", 0, None, "hey jarvis")
        await conn._handle_audio(_speech_chunk(), None)   # no trailing silence
        await conn._handle_stop(False)                    # device closed the stream
        await conn._finish_task

    _run(scenario())
    assert handled == [("what time is it", "voice:kitchen")]
    assert len(stt.calls) == 1


def test_handler_exception_sends_error_and_run_end():
    ch, conn, timeline, _, _ = make_channel()

    async def handler(text, sender):
        raise RuntimeError("kaboom")

    ch._handler = handler
    _run(_utterance(conn, [_speech_chunk(), _silence_chunk()]))
    events = _events(timeline)
    assert events[-2][0] == ET.VOICE_ASSISTANT_ERROR
    assert events[-1][0] == ET.VOICE_ASSISTANT_RUN_END


def test_setup_subscribes_and_enforces_wake_word():
    ch, conn, timeline, _, _ = make_channel()
    client = FakeAPIClient("10.0.0.5", 6053, None, timeline=timeline)
    _run(conn._setup(client))
    assert set(client.callbacks) == {"start", "stop", "audio"}
    assert client.callbacks["audio"] is not None       # API-audio subscription
    assert client.set_config_calls == [["hey_jarvis"]]


def test_setup_leaves_wake_word_when_already_active():
    ch, conn, timeline, _, _ = make_channel()
    client = FakeAPIClient(
        "10.0.0.5", 6053, None, timeline=timeline, active_wake_words=("hey_jarvis",)
    )
    _run(conn._setup(client))
    assert client.set_config_calls == []


def test_start_requires_bind():
    ch = VoicePEChannel(
        [VoicePEDeviceConfig(name="kitchen", host="10.0.0.5")],
        client_factory=lambda h, p, k: FakeAPIClient(h, p, k),
    )

    async def handler(text, sender):
        return None

    with pytest.raises(RuntimeError):
        ch.start(handler)
