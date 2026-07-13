import hashlib
import re
from memory.cache import FridayCache
from memory.store import FridayStore


class ContextBuilder:
    """Decides what memory context to inject for each user query.

    This is where we save tokens. Not every request needs full context.

    Privacy: callers pass include_private=False when the prompt is bound for a
    cloud provider — private memory (preferences, personal facts, summaries,
    device state) is then withheld entirely so it never leaves the device.
    """

    # Simple keyword-based routing. Upgrade to a local classifier later.
    DEVICE_KEYWORDS = {"turn", "switch", "light", "lamp", "thermostat", "lock", "door",
                       "garage", "fan", "blinds", "espresso", "coffee", "temperature"}
    PERSONAL_KEYWORDS = {"my", "i", "me", "prefer", "usually", "favorite", "always"}
    HISTORY_KEYWORDS = {"yesterday", "last time", "remember", "earlier", "before",
                        "did i", "what was", "when did"}

    # Queries whose answers change with time or device state must not be
    # served from the response cache.
    VOLATILE_KEYWORDS = {"time", "today", "now", "weather", "temperature", "date",
                         "locked", "unlocked", "on", "off", "status"}

    # Facts guessed by the extractor start at 0.4 — only assert facts we're
    # reasonably sure of.
    MIN_INJECT_CONFIDENCE = 0.6

    def __init__(self, cache: FridayCache, store: FridayStore):
        self.cache = cache
        self.store = store

    def query_fingerprint(self, query: str) -> str:
        """Normalize query for response caching."""
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _tokens(self, msg_lower: str) -> set:
        return set(re.findall(r"[a-z0-9']+", msg_lower))

    def is_cacheable(self, user_message: str) -> bool:
        """Whether a response to this message may be cached/served from cache."""
        return not (self.VOLATILE_KEYWORDS & self._tokens(user_message.lower()))

    def build_context(self, user_message: str, include_private: bool = True) -> dict:
        """Returns context dict to inject into the system prompt.

        Returns:
            {
                "cached_response": str or None,  # If set, skip the LLM entirely
                "device_state": dict or None,
                "preferences": dict or None,
                "personal": dict or None,
                "summaries": list or None,
                "relevant_facts": list or None,
                "retrieved_fact_keys": list,
            }
        """
        msg_lower = user_message.lower()
        tokens = self._tokens(msg_lower)
        context = {"retrieved_fact_keys": []}

        # 1. Check response cache first — can we skip the LLM entirely?
        #    Never replay answers to time/state-dependent questions.
        if self.is_cacheable(user_message):
            fp = self.query_fingerprint(user_message)
            cached = self.cache.get_cached_response(fp)
            if cached:
                context["cached_response"] = cached
                return context

        # Private memory never rides along on cloud-bound prompts.
        if not include_private:
            return context

        # 2. Conditionally include device state
        if self.DEVICE_KEYWORDS & tokens:
            device_state = self.cache.device_state.fetch("all_devices")
            if device_state:
                context["device_state"] = device_state

        retrieved_keys = []

        # 3. Conditionally include personal preferences
        if self.PERSONAL_KEYWORDS & tokens:
            prefs = self.store.recall_by_category(
                "preference", min_confidence=self.MIN_INJECT_CONFIDENCE)
            personal = self.store.recall_by_category(
                "personal", min_confidence=self.MIN_INJECT_CONFIDENCE)
            context["preferences"] = prefs
            context["personal"] = personal
            retrieved_keys.extend(prefs.keys())
            retrieved_keys.extend(personal.keys())

        # 4. Conditionally include history/summaries
        if self.HISTORY_KEYWORDS & tokens or any(kw in msg_lower for kw in self.HISTORY_KEYWORDS):
            context["summaries"] = self.store.get_summaries(n=3)
            facts = self.store.search_facts(
                msg_lower, min_confidence=self.MIN_INJECT_CONFIDENCE)
            context["relevant_facts"] = facts
            retrieved_keys.extend(key for key, _, _ in facts)

        context["retrieved_fact_keys"] = retrieved_keys
        return context

    def format_system_prompt(self, base_prompt: str, context: dict) -> str:
        """Append relevant context to the base system/personality prompt."""
        parts = [base_prompt]

        if context.get("device_state"):
            parts.append(f"\n<device_state>\n{context['device_state']}\n</device_state>")

        if context.get("preferences"):
            prefs = "\n".join(f"- {k}: {v}" for k, v in context["preferences"].items())
            parts.append(f"\n<user_preferences>\n{prefs}\n</user_preferences>")

        if context.get("personal"):
            personal = "\n".join(f"- {k}: {v}" for k, v in context["personal"].items())
            parts.append(f"\n<personal_info>\n{personal}\n</personal_info>")

        if context.get("summaries"):
            summaries = "\n".join(context["summaries"])
            parts.append(f"\n<conversation_history_summary>\n{summaries}\n</conversation_history_summary>")

        if context.get("relevant_facts"):
            facts = "\n".join(f"- {k}: {v}" for k, v, _ in context["relevant_facts"])
            parts.append(f"\n<relevant_memories>\n{facts}\n</relevant_memories>")

        return "\n".join(parts)
