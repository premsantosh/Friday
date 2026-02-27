import hashlib
import re
from typing import Optional
from memory.cache import FridayCache
from memory.store import FridayStore

class ContextBuilder:
    """Decides what memory context to inject for each user query.
    
    This is where we save tokens. Not every request needs full context.
    """
    
    # Simple keyword-based routing. Upgrade to a local classifier later.
    DEVICE_KEYWORDS = {"turn", "switch", "light", "lamp", "thermostat", "lock", "door", 
                       "garage", "fan", "blinds", "espresso", "coffee", "temperature"}
    PERSONAL_KEYWORDS = {"my", "i", "me", "prefer", "usually", "favorite", "always"}
    HISTORY_KEYWORDS = {"yesterday", "last time", "remember", "earlier", "before", 
                        "did i", "what was", "when did"}
    
    def __init__(self, cache: FridayCache, store: FridayStore):
        self.cache = cache
        self.store = store
    
    def query_fingerprint(self, query: str) -> str:
        """Normalize query for response caching."""
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def build_context(self, user_message: str) -> dict:
        """Returns context dict to inject into the system prompt.
        
        Returns:
            {
                "cached_response": str or None,  # If set, skip Claude entirely
                "device_state": dict or None,
                "preferences": dict or None,
                "conversation_history": list or None,
                "summaries": list or None,
                "relevant_facts": list or None,
            }
        """
        msg_lower = user_message.lower()
        context = {}
        
        # 1. Check response cache first — can we skip Claude entirely?
        fp = self.query_fingerprint(user_message)
        cached = self.cache.get_cached_response(fp)
        if cached:
            context["cached_response"] = cached
            return context
        
        # 2. Always include recent conversation turns (last 3 for voice continuity)
        context["conversation_history"] = self.store.get_recent_turns(n=3)
        
        # 3. Conditionally include device state
        if self.DEVICE_KEYWORDS & set(msg_lower.split()):
            device_state = self.cache.device_state.fetch("all_devices")
            if device_state:
                context["device_state"] = device_state
        
        retrieved_keys = []

        # 4. Conditionally include personal preferences
        if self.PERSONAL_KEYWORDS & set(msg_lower.split()):
            prefs = self.store.recall_by_category("preference")
            personal = self.store.recall_by_category("personal")
            context["preferences"] = prefs
            context["personal"] = personal
            retrieved_keys.extend(prefs.keys())
            retrieved_keys.extend(personal.keys())

        # 5. Conditionally include history/summaries
        if self.HISTORY_KEYWORDS & set(msg_lower.split()):
            context["summaries"] = self.store.get_summaries(n=3)
            facts = self.store.search_facts(msg_lower)
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
            parts.append(f"\n<user_preferences>\n{context['preferences']}\n</user_preferences>")
        
        if context.get("personal"):
            parts.append(f"\n<personal_info>\n{context['personal']}\n</personal_info>")
        
        if context.get("summaries"):
            summaries = "\n".join(context["summaries"])
            parts.append(f"\n<conversation_history_summary>\n{summaries}\n</conversation_history_summary>")
        
        return "\n".join(parts)
