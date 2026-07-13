import json
import requests

_EXTRACTION_PROMPT = """You are a fact extractor for a personal assistant. Given a conversation exchange, extract facts worth remembering long-term about the user.

Categories:
- "preference": things the user likes/dislikes/wants done a certain way
- "personal": identity facts (name, location, job, family)
- "routine": habitual behaviors (e.g. "usually wakes at 7am")

Only extract facts clearly about the user and worth remembering across sessions.
Ignore commands, one-off requests, and general questions.

Return a JSON array only. If nothing worth storing, return [].
Example: [{{"key": "coffee_order", "value": "oat milk flat white", "category": "preference"}}]

User: {user_msg}
Assistant: {assistant_msg}"""

# A correction must LEAD with the correction ("No, I take oat milk") or use an
# explicit phrase. A bare "no"/"actually" mid-sentence is too weak a signal to
# penalize stored facts — it deleted memories spuriously.
_CORRECTION_STARTS = {"no", "nope", "wrong", "incorrect", "actually"}
_CORRECTION_PHRASES = {
    "that's wrong", "thats wrong", "that's not right", "thats not right",
    "that's incorrect", "thats incorrect", "you're wrong", "youre wrong",
    "you're mistaken", "youre mistaken", "not what i said",
}

_EXPLICIT_MARKERS = {
    "my ", "i am ", "i'm ", "i prefer", "i always", "i usually",
    "i live", "i work", "i like", "i love", "i hate", "i need",
}


class FactExtractor:
    """Uses a local Ollama model to extract facts from conversation turns."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3:mini"):
        self.base_url = base_url
        self.model = model

    def extract(self, user_msg: str, assistant_msg: str) -> list[dict]:
        """Extract storable facts from an exchange.

        Returns list of {"key", "value", "category", "confidence"}.
        Returns [] on any failure — never raises.
        """
        prompt = _EXTRACTION_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            facts = json.loads(raw[start:end])
            valid = [
                f for f in facts
                if isinstance(f, dict) and "key" in f and "value" in f and "category" in f
            ]
        except Exception:
            return []

        confidence = 0.9 if self._is_explicit(user_msg) else 0.4
        for f in valid:
            f["confidence"] = confidence
        return valid

    def is_correction(self, user_msg: str) -> bool:
        """Heuristic: did the user correct the assistant?"""
        lower = user_msg.lower()
        first = lower.split()[0].strip(",.!?") if lower.split() else ""
        return first in _CORRECTION_STARTS or any(p in lower for p in _CORRECTION_PHRASES)

    def _is_explicit(self, user_msg: str) -> bool:
        lower = user_msg.lower()
        return any(marker in lower for marker in _EXPLICIT_MARKERS)
