from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    published_date: Optional[str] = None   # when the source exposes it (recency)
    score: Optional[float] = None          # provider relevance score, if any


class SearchProvider(ABC):
    """Fetches web search results for a query."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5,
               include_domains: Optional[List[str]] = None) -> List[SearchResult]:
        """Run a web search. `include_domains` restricts results to those sites
        (e.g. ["reddit.com"]) when the provider supports it. Returns [] on failure."""
        pass


class TavilySearchProvider(SearchProvider):
    """AI-optimized web search via the Tavily API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5,
               include_domains: Optional[List[str]] = None) -> List[SearchResult]:
        """Search via Tavily. Returns [] on any failure."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            kwargs = {"max_results": max_results}
            if include_domains:
                kwargs["include_domains"] = include_domains
            response = client.search(query, **kwargs)
            return [
                SearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    url=r.get("url", ""),
                    published_date=r.get("published_date"),
                    score=r.get("score"),
                )
                for r in response.get("results", [])
            ]
        except Exception:
            return []
