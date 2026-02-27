from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchProvider(ABC):
    """Fetches web search results for a query."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Run a web search. Returns [] on any failure."""
        pass


class TavilySearchProvider(SearchProvider):
    """AI-optimized web search via the Tavily API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search via Tavily. Returns [] on any failure."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query, max_results=max_results)
            return [
                SearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    url=r.get("url", ""),
                )
                for r in response.get("results", [])
            ]
        except Exception:
            return []
