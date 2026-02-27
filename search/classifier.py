import json
from abc import ABC, abstractmethod

import requests

_CLASSIFICATION_PROMPT = """You are a search-need classifier. Given a user message to a voice assistant, decide whether answering it requires up-to-date information from the internet.

Return JSON only: {{"needs_search": true/false, "search_query": "optimized search query or empty string"}}

needs_search = true when the question is about:
- Current events, news, weather, sports scores
- Real-time data (stock prices, flight status, etc.)
- Facts the assistant might not know or that change over time
- Specific people, companies, or products that need current info

needs_search = false when:
- The user is making a command (turn on lights, set timer, etc.)
- The question is conversational or personal (how are you, tell me a joke)
- The question is about well-known, static facts (what is 2+2, what colour is the sky)
- The user is adjusting assistant settings (more sarcasm, be serious)

User message: {user_input}"""


class SearchClassifier(ABC):
    """Decides whether a user query needs a web search."""

    @abstractmethod
    def classify(self, user_input: str) -> tuple[bool, str]:
        """Classify whether the input needs a web search.

        Returns (needs_search, optimized_search_query).
        Returns (False, "") on any failure.
        """
        pass


class OllamaSearchClassifier(SearchClassifier):
    """Uses a local Ollama model to classify search need."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3:mini"):
        self.base_url = base_url
        self.model = model

    def classify(self, user_input: str) -> tuple[bool, str]:
        """Classify via Ollama. Returns (False, '') on any failure."""
        prompt = _CLASSIFICATION_PROMPT.format(user_input=user_input)
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return False, ""
            data = json.loads(raw[start:end])
            needs = data.get("needs_search", False)
            query = data.get("search_query", "").strip()
            if needs and query:
                return True, query
            return False, ""
        except Exception:
            return False, ""
