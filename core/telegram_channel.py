"""
Two-way Telegram channel.

Lets you chat with Friday from your phone over the Telegram Bot API — which
needs no phone number, SMS, or captcha: you create a bot with @BotFather, get a
token, and that's it. A daemon thread long-polls getUpdates for incoming
messages, hands each to an async handler (normally `VoiceAssistant.process_input`),
and replies with sendMessage.

Design:
  * One daemon thread owning its own asyncio loop; a stop Event; never raises
    out of the loop (logs + backs off).
  * Security: only chat IDs on the allowlist get a response. Unlike a phone
    number, you don't know your Telegram chat ID until you message the bot — so
    when the allowlist is empty the bot replies with the chat ID and how to
    authorise it, but does NOT act on the message (fail closed).
  * HTTP is injectable (`poster`/`getter`) so tests never touch the network.

getUpdates uses an `offset` cursor to acknowledge processed updates; we advance
it past any backlog on startup so a restart doesn't replay stale commands.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Awaitable, Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# handler(text, chat_id) -> reply text (or None/"" to stay silent).
Handler = Callable[[str, str], Awaitable[Optional[str]]]


class TelegramChannel:
    def __init__(
        self,
        token: str,
        allowed_chat_ids: Optional[Iterable] = None,
        long_poll_seconds: int = 20,
        poster: Optional[Callable] = None,
        getter: Optional[Callable] = None,
    ):
        self._token = token
        self.api = f"https://api.telegram.org/bot{token}"
        # Chat IDs are numeric but compared as strings for consistency.
        self.allowed_chat_ids = {str(c).strip() for c in (allowed_chat_ids or ()) if str(c).strip()}
        self.long_poll_seconds = max(0, int(long_poll_seconds))
        self._poster = poster  # callable(url, json=..., timeout=...) -> resp
        self._getter = getter  # callable(url, params=..., timeout=...) -> resp

        self._handler: Optional[Handler] = None
        self._offset: Optional[int] = None
        self._skip_backlog = True
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Optional research-substrate hooks (set by main.py; default off keeps
        # behavior identical to before they existed):
        #   feedback_provider(chat_id, reply) -> reply_markup dict or None,
        #     consulted per reply to attach an inline keyboard (👍/👎).
        #   on_callback(chat_id, callback_data) -> None, invoked for button taps
        #     from allowlisted chats.
        self.feedback_provider: Optional[Callable[[str, str], Optional[dict]]] = None
        self.on_callback: Optional[Callable[[str, str], None]] = None

    # ------------------------------------------------------------------ config
    @classmethod
    def from_env(cls) -> Optional["TelegramChannel"]:
        """Build from env, or None when no bot token is configured.

        TELEGRAM_BOT_TOKEN comes from @BotFather. TELEGRAM_ALLOWED_CHAT_IDS is a
        comma-separated allowlist of numeric chat IDs; leave it unset on first
        run and the bot will tell you your chat ID so you can fill it in.
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            return None
        raw_allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        allowed = [c.strip() for c in raw_allowed.split(",") if c.strip()]
        if not allowed:
            logger.warning(
                "Telegram channel: no TELEGRAM_ALLOWED_CHAT_IDS set; the bot will "
                "reply with the chat ID to authorise but won't act on messages yet."
            )
        long_poll = int(os.getenv("TELEGRAM_LONG_POLL_SECONDS", "20"))
        return cls(token, allowed_chat_ids=allowed, long_poll_seconds=long_poll)

    # ------------------------------------------------------------------ outbound
    def send(self, message: str, chat_id, reply_markup: Optional[dict] = None) -> bool:
        """Send `message` to `chat_id`. Returns success; never raises."""
        try:
            poster = self._poster
            if poster is None:
                import requests
                poster = requests.post
            payload = {"chat_id": chat_id, "text": message}
            if reply_markup:
                payload["reply_markup"] = reply_markup
            resp = poster(
                f"{self.api}/sendMessage",
                json=payload,
                timeout=15,
            )
            status = int(getattr(resp, "status_code", 200))
            if not (200 <= status < 300):
                logger.warning("Telegram sendMessage to %s returned HTTP %s", chat_id, status)
            return 200 <= status < 300
        except Exception:
            logger.warning("Telegram send failed.", exc_info=True)
            return False

    # ------------------------------------------------------------------ lifecycle
    def start(self, handler: Handler) -> None:
        """Begin long-polling for inbound messages, dispatching each to `handler`."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._handler = handler
        self._skip_backlog = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="telegram-channel")
        self._thread.start()
        logger.info(
            "TelegramChannel started (long_poll=%ss, %d allowed chat(s)).",
            self.long_poll_seconds, len(self.allowed_chat_ids),
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
            loop.run_until_complete(self._async_loop())
        finally:
            loop.close()

    async def _async_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception:
                logger.exception("Telegram poll failed — backing off.")
                # Back off so a persistent error (bad token, no network) doesn't
                # hammer the API.
                slept = 0.0
                while slept < 5.0 and not self._stop_event.is_set():
                    await asyncio.sleep(0.25)
                    slept += 0.25

    async def _poll_once(self) -> None:
        getter = self._getter
        if getter is None:
            import requests
            getter = requests.get
        # The startup backlog drain must be non-blocking (timeout=0): a long poll
        # here would wait and scoop up the user's first *live* message into the
        # batch we discard. After the drain, switch to normal long polling.
        long_poll = 0 if self._skip_backlog else self.long_poll_seconds
        params = {"timeout": long_poll}
        if self._offset is not None:
            params["offset"] = self._offset
        resp = getter(
            f"{self.api}/getUpdates",
            params=params,
            timeout=long_poll + 15,
        )
        status = int(getattr(resp, "status_code", 200))
        if status == 409:
            # Telegram allows only one getUpdates poller per bot at a time.
            logger.warning(
                "Telegram getUpdates conflict (HTTP 409): another Friday instance "
                "is already polling this bot. Run only one — stop the duplicate "
                "(`pkill -f main.py`) and restart."
            )
            return
        if not (200 <= status < 300):
            logger.warning("Telegram getUpdates returned HTTP %s", status)
            return
        try:
            payload = resp.json()
        except Exception:
            logger.warning("Telegram getUpdates returned non-JSON body.", exc_info=True)
            return
        if not isinstance(payload, dict) or not payload.get("ok"):
            logger.warning("Telegram getUpdates not ok (check bot token).")
            return

        result = payload.get("result") or []
        # Advance the cursor past every update returned (including non-text ones)
        # so they are acknowledged and never re-delivered.
        if result:
            self._offset = max(int(u.get("update_id", 0)) for u in result) + 1

        # On the first poll after start, drop whatever was queued while we were
        # down rather than replaying potentially stale commands.
        if self._skip_backlog:
            self._skip_backlog = False
            if result:
                logger.info("Telegram: skipped %d backlog update(s) on startup.", len(result))
            return

        for chat_id, text in self._extract_messages(result):
            if not self.allowed_chat_ids:
                # No allowlist yet — help the user bootstrap, but don't act.
                self.send(
                    "Friday isn't authorised for this chat yet. Add this chat ID to "
                    f"TELEGRAM_ALLOWED_CHAT_IDS and restart:\n{chat_id}",
                    chat_id,
                )
                logger.warning(
                    "Telegram message from chat %s but no allowlist configured.", chat_id
                )
                continue
            if chat_id not in self.allowed_chat_ids:
                logger.info("Ignoring Telegram message from unauthorised chat %s", chat_id)
                continue
            logger.info("Telegram inbound from %s: %r", chat_id, text)
            await self._dispatch(chat_id, text)

        for callback_id, chat_id, data in self._extract_callbacks(result):
            # Always answer the callback so the client stops its spinner, even
            # for unauthorised chats (answering leaks nothing).
            self._answer_callback(callback_id)
            if chat_id not in self.allowed_chat_ids:
                logger.info("Ignoring Telegram callback from unauthorised chat %s", chat_id)
                continue
            if self.on_callback is None:
                continue
            try:
                self.on_callback(chat_id, data)
            except Exception:
                logger.warning("Telegram callback handler raised.", exc_info=True)

    @staticmethod
    def _extract_messages(result) -> List[Tuple[str, str]]:
        """Pull (chat_id, text) pairs out of a getUpdates result list.

        Only plain text messages are handled; non-text updates (stickers,
        callbacks, joins, edited-without-text) are skipped. chat_id is stringified
        for stable allowlist comparison.
        """
        out: List[Tuple[str, str]] = []
        if not isinstance(result, list):
            return out
        for u in result:
            if not isinstance(u, dict):
                continue
            msg = u.get("message") or u.get("edited_message") or {}
            text = msg.get("text")
            if not text or not text.strip():
                continue
            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue
            out.append((str(chat_id), text.strip()))
        return out

    @staticmethod
    def _extract_callbacks(result) -> List[Tuple[str, str, str]]:
        """Pull (callback_query_id, chat_id, data) out of a getUpdates result list.

        Callback queries arrive when a user taps an inline keyboard button
        (e.g. the 👍/👎 feedback buttons). The offset cursor already
        acknowledges them; this makes them actionable.
        """
        out: List[Tuple[str, str, str]] = []
        if not isinstance(result, list):
            return out
        for u in result:
            if not isinstance(u, dict):
                continue
            cq = u.get("callback_query")
            if not isinstance(cq, dict):
                continue
            data = cq.get("data")
            callback_id = cq.get("id")
            chat = (cq.get("message") or {}).get("chat") or {}
            chat_id = chat.get("id")
            if not data or callback_id is None or chat_id is None:
                continue
            out.append((str(callback_id), str(chat_id), str(data)))
        return out

    def _answer_callback(self, callback_id: str) -> None:
        """Acknowledge a callback query (stops the client spinner). Never raises."""
        try:
            poster = self._poster
            if poster is None:
                import requests
                poster = requests.post
            poster(
                f"{self.api}/answerCallbackQuery",
                json={"callback_query_id": callback_id},
                timeout=15,
            )
        except Exception:
            logger.warning("Telegram answerCallbackQuery failed.", exc_info=True)

    async def _dispatch(self, chat_id: str, text: str) -> None:
        assert self._handler is not None
        try:
            reply = await self._handler(text, chat_id)
        except Exception:
            logger.exception("Telegram handler raised for message from %s", chat_id)
            self.send("I ran into an error handling that, sir.", chat_id)
            return
        if reply and reply.strip():
            logger.info("Telegram reply to %s: %r", chat_id, reply[:80])
            reply_markup = None
            if self.feedback_provider is not None:
                try:
                    reply_markup = self.feedback_provider(chat_id, reply)
                except Exception:
                    logger.warning("Telegram feedback provider raised.", exc_info=True)
            self.send(reply, chat_id, reply_markup=reply_markup)
