"""
Conversation summarizer — runs entirely on-device.

Old raw turns are condensed into summaries by the local Ollama model so
long-term conversational memory never leaves the machine. Summaries are
injected back into prompts by the ContextBuilder.
"""

import logging

import requests

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = (
    "Summarize this conversation concisely. Focus on: facts learned about the "
    "user, decisions made, preferences expressed, and any ongoing topics. Be brief."
)


class ConversationSummarizer:
    """Summarizes conversation history to save tokens and disk space."""

    def __init__(self, store, base_url: str = "http://localhost:11434",
                 model: str = "llama3.1"):
        self.store = store
        self.base_url = base_url
        self.model = model

    def summarize_and_prune(self, max_turns_before_summary: int = 50,
                            keep_recent: int = 10):
        """Summarize the oldest turns and delete exactly those rows.

        Keeps the most recent `keep_recent` turns raw for short-term recall.
        No-op (never raises) if there aren't enough turns or Ollama is down.
        """
        total = self.store.count_turns()
        if total < max_turns_before_summary:
            return

        turns = self.store.get_oldest_turns(total - keep_recent)
        if not turns:
            return

        conversation_text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SUMMARY_PROMPT},
                        {"role": "user", "content": conversation_text},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 500},
                },
                timeout=60,
            )
            resp.raise_for_status()
            summary = resp.json()["message"]["content"].strip()
        except Exception:
            logger.warning("Local summarization failed; keeping raw turns", exc_info=True)
            return

        if not summary:
            return

        ids = [t["id"] for t in turns]
        self.store.save_summary(summary, start_id=ids[0], end_id=ids[-1])
        self.store.delete_turns(ids)
