"""
Voice PE satellite channel.

Home Assistant Voice Preview Edition pucks act as room microphones for Friday —
no Home Assistant server involved. Friday connects straight to each device over
the ESPHome native API (aioesphomeapi) and registers itself as the device's
voice-assistant pipeline: the puck does wake-word detection ("Hey Jarvis") and
streams raw mic audio; Friday does STT, thinks, and speaks the reply on the
Mac's speakers. Nothing plays on the puck's own speaker — we never send the
TTS_END event, which is what would carry a media URL for it to play.

Design (mirrors core/telegram_channel.py):
  * from_env() -> None when VOICE_PE_DEVICES is unset.
  * One daemon thread owning one asyncio loop shared by every satellite;
    blocking work (Whisper, speaker playback) runs in the default executor so
    one room's transcription doesn't stall another room's audio stream.
  * bind() hands the channel the assistant's stt/speak/has_active before
    start(handler); handler is the same `async (text, sender) -> reply` used by
    _attach_listener, with sender "voice:<device name>" so each room gets its
    own multi-turn session.
  * The API client is injectable (client_factory) so tests never touch the
    network. Never raises out of the loop; a lost device reconnects with
    backoff.

Per-utterance event flow (_SatelliteConnection._finish_utterance): the device
opens a run after the wake word; we answer RUN_START/STT_START, buffer 16 kHz
int16 audio with an RMS end-of-speech gate, transcribe, hand the text to the
brain, speak the reply on the Mac *before* INTENT_END (once INTENT_END lands
with continue_conversation the puck re-opens its mic, and its echo canceller
can't remove audio coming from our speakers), and set continue_conversation
from the session state so mid-dialogue turns need no wake word.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, List, Optional

import numpy as np

try:
    from aioesphomeapi import APIClient, VoiceAssistantEventType, ZERO_NOISE_PSK
    from aioesphomeapi.core import (
        InvalidEncryptionKeyAPIError,
        RequiresEncryptionAPIError,
    )
    HAVE_AIOESPHOMEAPI = True
except ImportError:
    HAVE_AIOESPHOMEAPI = False
    APIClient = None
    VoiceAssistantEventType = None
    ZERO_NOISE_PSK = ""

    class RequiresEncryptionAPIError(Exception):
        pass

    class InvalidEncryptionKeyAPIError(Exception):
        pass

logger = logging.getLogger(__name__)

# handler(text, sender) -> reply text (or None/"" to stay silent).
Handler = Callable[[str, str], Awaitable[Optional[str]]]

# The ESPHome voice pipeline streams 16-bit LE PCM at 16 kHz.
_SAMPLE_RATE = 16000
_DEFAULT_PORT = 6053


def _env_suffix(name: str) -> str:
    """Device name -> per-device env-var suffix ("living-room" -> LIVING_ROOM)."""
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _wake_word_id(phrase: str) -> str:
    """Wake-word phrase -> firmware model id ("Hey Jarvis" -> hey_jarvis)."""
    return re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower()).strip("_")


def _default_client_factory(host: str, port: int, noise_psk: Optional[str]):
    return APIClient(host, port, None, noise_psk=noise_psk)


@dataclass
class VoicePEDeviceConfig:
    name: str
    host: str
    port: int = _DEFAULT_PORT
    noise_psk: Optional[str] = None


class _SatelliteConnection:
    """One Voice PE device on the channel's shared asyncio loop.

    Owns the API client, the connect/reconnect loop, and the state of the one
    in-flight utterance pipeline. Fully independent of the other devices.
    """

    def __init__(self, cfg: VoicePEDeviceConfig, channel: "VoicePEChannel"):
        self.cfg = cfg
        self.channel = channel
        self.user_id = f"voice:{cfg.name}"
        self._client = None
        # A fresh (never HA-adopted) device accepts plaintext or the well-known
        # all-zeros PSK; try both when no key is configured.
        self._psk_candidates: List[Optional[str]] = (
            [cfg.noise_psk] if cfg.noise_psk else [None, ZERO_NOISE_PSK]
        )
        self._psk_index = 0
        self._disconnected: Optional[asyncio.Event] = None
        self._collecting = False
        self._chunks: List[bytes] = []
        self._speech_detected = False
        self._silence_samples = 0
        self._total_samples = 0
        self._conversation_id = ""
        self._finish_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------- connection
    async def run(self) -> None:
        """Connect-and-reconnect loop; returns only when the channel stops."""
        backoff = 1.0
        stop = self.channel._stop_async
        while not stop.is_set():
            self._disconnected = asyncio.Event()
            client = None
            try:
                client = self.channel._client_factory(
                    self.cfg.host, self.cfg.port, self._psk_candidates[self._psk_index]
                )
                self._client = client
                await client.connect(login=True, on_stop=self._on_conn_stop)
                await self._setup(client)
            except (RequiresEncryptionAPIError, InvalidEncryptionKeyAPIError) as e:
                await self._close(client)
                if self._psk_index + 1 < len(self._psk_candidates):
                    self._psk_index += 1
                    logger.info(
                        "Voice PE [%s]: %s — retrying with the well-known zero PSK.",
                        self.cfg.name, type(e).__name__,
                    )
                    continue
                logger.warning(
                    "Voice PE [%s]: encryption key rejected (%s). If this device "
                    "was ever adopted by Home Assistant, put its key in "
                    "VOICE_PE_NOISE_PSK or VOICE_PE_NOISE_PSK_%s — or factory-"
                    "reset the device. Retrying in %.0fs.",
                    self.cfg.name, type(e).__name__, _env_suffix(self.cfg.name), backoff,
                )
                await self._interruptible_sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            except asyncio.CancelledError:
                await self._close(client)
                raise
            except Exception:
                await self._close(client)
                logger.warning(
                    "Voice PE [%s]: connect to %s:%d failed — retrying in %.0fs.",
                    self.cfg.name, self.cfg.host, self.cfg.port, backoff, exc_info=True,
                )
                await self._interruptible_sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            logger.info(
                "Voice PE [%s] connected (%s:%d).",
                self.cfg.name, self.cfg.host, self.cfg.port,
            )
            backoff = 1.0
            await self._wait_stop_or_disconnect()
            self._reset_pipeline()
            await self._close(client)
            if not stop.is_set():
                logger.warning("Voice PE [%s] disconnected — reconnecting.", self.cfg.name)

    async def _setup(self, client) -> None:
        """Register as the voice pipeline and enforce the wake word.

        Runs on every (re)connect — subscriptions die with the connection.
        Passing handle_audio selects API-audio mode (chunks over TCP, no UDP).
        """
        client.subscribe_voice_assistant(
            handle_start=self._handle_start,
            handle_stop=self._handle_stop,
            handle_audio=self._handle_audio,
        )
        await self._enforce_wake_word(client)

    async def _enforce_wake_word(self, client) -> None:
        wanted = self.channel.wake_word_id
        try:
            cfg = await client.get_voice_assistant_configuration(timeout=5.0)
        except Exception:
            logger.warning(
                "Voice PE [%s]: could not read wake-word configuration.",
                self.cfg.name, exc_info=True,
            )
            return
        available = [w for w in getattr(cfg, "available_wake_words", ())]
        ids = {w.id for w in available}
        if wanted not in ids:
            # Fall back to matching the spoken phrase in case ids differ by firmware.
            by_phrase = {
                _wake_word_id(getattr(w, "wake_word", "")): w.id for w in available
            }
            found = by_phrase.get(wanted)
            if found is None:
                logger.warning(
                    "Voice PE [%s]: wake word %r not on the device (has: %s) — "
                    "leaving its configuration alone.",
                    self.cfg.name, wanted, sorted(ids),
                )
                return
            wanted = found
        if list(getattr(cfg, "active_wake_words", ())) != [wanted]:
            await client.set_voice_assistant_configuration(active_wake_words=[wanted])
            logger.info("Voice PE [%s]: active wake word set to %r.", self.cfg.name, wanted)

    async def _on_conn_stop(self, expected_disconnect: bool) -> None:
        if self._disconnected is not None:
            self._disconnected.set()

    async def _wait_stop_or_disconnect(self) -> None:
        stop_t = asyncio.create_task(self.channel._stop_async.wait())
        disc_t = asyncio.create_task(self._disconnected.wait())
        _, pending = await asyncio.wait({stop_t, disc_t}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()

    async def _interruptible_sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.channel._stop_async.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _close(self, client) -> None:
        try:
            await client.disconnect()
        except Exception:
            pass
        self._client = None

    # --------------------------------------------------------------- pipeline
    def _reset_pipeline(self) -> None:
        if self._finish_task is not None and not self._finish_task.done():
            self._finish_task.cancel()
        self._finish_task = None
        self._collecting = False
        self._chunks = []
        self._speech_detected = False
        self._silence_samples = 0
        self._total_samples = 0

    async def _handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        """Device opened a run (wake word heard, or a continue-conversation
        turn — then the USE_WAKE_WORD flag is clear and there's no phrase)."""
        self._reset_pipeline()
        self._conversation_id = conversation_id or ""
        self._collecting = True
        self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START)
        self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_START)
        return 0  # API-audio mode: no UDP port

    async def _handle_audio(self, data: bytes, data2: Optional[bytes] = None) -> None:
        # data is mic channel 0 (processed); data2 is the second channel — unused.
        if not self._collecting or not data:
            return
        self._chunks.append(data)
        samples = np.frombuffer(data, dtype=np.int16)
        n = len(samples)
        self._total_samples += n
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        ch = self.channel
        if rms > ch.silence_threshold:
            self._speech_detected = True
            self._silence_samples = 0
        elif self._speech_detected:
            self._silence_samples += n
            if self._silence_samples >= ch.silence_seconds * _SAMPLE_RATE:
                self._begin_finish()
                return
        if not self._speech_detected and self._total_samples >= ch.no_speech_seconds * _SAMPLE_RATE:
            self._begin_finish()
            return
        if self._total_samples >= ch.max_utterance_seconds * _SAMPLE_RATE:
            self._begin_finish()

    async def _handle_stop(self, abort: bool) -> None:
        if abort:
            self._reset_pipeline()
            return
        # Device ended the audio stream itself — treat as end-of-speech.
        if self._collecting:
            self._begin_finish()

    def _begin_finish(self) -> None:
        self._collecting = False
        self._finish_task = asyncio.get_running_loop().create_task(self._finish_utterance())

    async def _finish_utterance(self) -> None:
        ch = self.channel
        try:
            if not self._speech_detected:
                # The Voice PE firmware special-cases this code: no error LED.
                self._send_error("stt-no-text-recognized", "No speech detected")
                return
            pcm = b"".join(self._chunks)
            self._chunks = []
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, ch._transcribe, audio)
            text = (text or "").strip()
            if not text:
                self._send_error("stt-no-text-recognized", "Could not transcribe speech")
                return
            # Firmware ignores an STT_END without non-empty text.
            self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_END, {"text": text})
            self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_START)
            logger.info("Voice PE [%s] heard: %r", self.cfg.name, text)
            reply = await ch._handler(text, self.user_id)
            if reply and reply.strip():
                # TTS_START drives the "replying" LED phase; without a TTS_END
                # (which would carry a URL) the puck plays nothing itself.
                self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": reply})
                await loop.run_in_executor(None, ch._speak_locked, reply)
            cont = "1" if ch._has_active(self.user_id) else "0"
            self._send_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END,
                {"conversation_id": self._conversation_id, "continue_conversation": cont},
            )
            self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Voice PE [%s] pipeline failed", self.cfg.name)
            self._send_error("pipeline-error", "Internal error handling the request")

    def _send_error(self, code: str, message: str) -> None:
        self._send_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
            {"code": code, "message": message},
        )
        self._send_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END)

    def _send_event(self, event_type, data: Optional[dict] = None) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.send_voice_assistant_event(event_type, data)
        except Exception:
            logger.warning(
                "Voice PE [%s]: failed to send %s.",
                self.cfg.name, getattr(event_type, "name", event_type), exc_info=True,
            )


class VoicePEChannel:
    def __init__(
        self,
        devices: Iterable[VoicePEDeviceConfig],
        wake_word: str = "Hey Jarvis",
        silence_seconds: float = 1.2,
        max_utterance_seconds: float = 15.0,
        no_speech_seconds: float = 5.0,
        silence_threshold: float = 0.01,
        client_factory: Optional[Callable] = None,
    ):
        if not HAVE_AIOESPHOMEAPI:
            raise ImportError(
                "aioesphomeapi is required for the Voice PE channel. "
                "Run: pip install 'aioesphomeapi>=45.6.2,<46' (needs Python >= 3.11)"
            )
        self.devices = list(devices)
        self.wake_word_id = _wake_word_id(wake_word)
        self.silence_seconds = silence_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.no_speech_seconds = no_speech_seconds
        self.silence_threshold = silence_threshold
        self._client_factory = client_factory or _default_client_factory

        self._stt = None            # STTProvider (.transcribe(audio, rate) -> str)
        self._speak: Optional[Callable[[str], None]] = None
        self._has_active: Callable[[str], bool] = lambda user_id: False
        self._handler: Optional[Handler] = None

        # Two rooms share one Whisper model and one set of speakers.
        self._stt_lock = threading.Lock()
        self._speaker_lock = threading.Lock()

        self._connections = [_SatelliteConnection(cfg, self) for cfg in self.devices]
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_async: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------ config
    @classmethod
    def from_env(cls) -> Optional["VoicePEChannel"]:
        """Build from env, or None when VOICE_PE_DEVICES is unset.

        VOICE_PE_DEVICES is `name=host[:port]`, comma-separated. Malformed
        entries are logged and skipped. See .env.example for the key options.
        """
        raw = os.getenv("VOICE_PE_DEVICES")
        if not raw:
            return None
        if not HAVE_AIOESPHOMEAPI:
            logger.warning(
                "VOICE_PE_DEVICES is set but aioesphomeapi is not installed — "
                "Voice PE channel disabled. Run: pip install -r requirements.txt "
                "(needs Python >= 3.11)."
            )
            return None
        global_psk = os.getenv("VOICE_PE_NOISE_PSK") or None
        devices: List[VoicePEDeviceConfig] = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, sep, addr = entry.partition("=")
            name, addr = name.strip(), addr.strip()
            if not sep or not name or not addr:
                logger.warning(
                    "Voice PE: skipping malformed VOICE_PE_DEVICES entry %r "
                    "(want name=host[:port]).", entry,
                )
                continue
            host, _, port_s = addr.partition(":")
            host = host.strip()
            port = _DEFAULT_PORT
            if port_s:
                try:
                    port = int(port_s)
                except ValueError:
                    logger.warning(
                        "Voice PE: skipping entry %r — bad port %r.", entry, port_s
                    )
                    continue
            psk = os.getenv(f"VOICE_PE_NOISE_PSK_{_env_suffix(name)}") or global_psk
            devices.append(VoicePEDeviceConfig(name=name, host=host, port=port, noise_psk=psk))
        if not devices:
            logger.warning(
                "VOICE_PE_DEVICES is set but no valid entries were parsed — "
                "Voice PE channel disabled."
            )
            return None
        return cls(
            devices,
            wake_word=os.getenv("VOICE_PE_WAKE_WORD", "Hey Jarvis"),
            silence_seconds=float(os.getenv("VOICE_PE_SILENCE_SECONDS", "1.2")),
            max_utterance_seconds=float(os.getenv("VOICE_PE_MAX_UTTERANCE_SECONDS", "15")),
        )

    # ------------------------------------------------------------------ wiring
    def bind(
        self,
        stt,
        speak: Callable[[str], None],
        has_active: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Hand over the assistant pieces the pipeline needs; call before start().

        `stt` is an STTProvider (needs .transcribe(audio, rate)); `speak` plays
        text on the Mac speakers (blocking); `has_active(user_id)` reports an
        open multi-turn session — None (sessions disabled) means dialogues
        simply never continue hands-free.
        """
        self._stt = stt
        self._speak = speak
        self._has_active = has_active or (lambda user_id: False)

    # ------------------------------------------------------------------ lifecycle
    def start(self, handler: Handler) -> None:
        """Connect to every satellite and begin handling utterances."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._stt is None or self._speak is None:
            raise RuntimeError("VoicePEChannel.bind() must be called before start().")
        self._handler = handler
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-pe-channel")
        self._thread.start()
        logger.info(
            "VoicePEChannel started (%d device(s): %s; wake word %r).",
            len(self.devices), ", ".join(d.name for d in self.devices), self.wake_word_id,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ------------------------------------------------------------------ internals
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()

    async def _async_main(self) -> None:
        self._stop_async = asyncio.Event()
        tasks = [asyncio.create_task(conn.run()) for conn in self._connections]
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
        self._stop_async.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        # Whisper isn't guaranteed thread-safe — serialize the (rare) case of
        # two rooms finishing an utterance at the same moment.
        with self._stt_lock:
            return self._stt.transcribe(audio, _SAMPLE_RATE)

    def _speak_locked(self, text: str) -> None:
        with self._speaker_lock:
            self._speak(text)
