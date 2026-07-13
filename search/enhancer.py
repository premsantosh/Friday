from typing import Optional

from search.classifier import SearchClassifier
from search.provider import SearchProvider


class SearchEnhancer:
    """Orchestrates search classification, fetching, and prompt formatting."""

    def __init__(self, classifier: SearchClassifier, provider: SearchProvider, max_results: int = 5):
        self.classifier = classifier
        self.provider = provider
        self.max_results = max_results

    def enhance(self, user_input: str) -> Optional[str]:
        """Return a search context block to append to the system prompt, or None.

        Returns None on any failure — never raises.
        """
        try:
            needs_search, query = self.classifier.classify(user_input)
            if not needs_search:
                return None

            # The query is the one payload that leaves the device on the
            # default local-first setup — scrub PII before it goes out.
            from core.harness import redact_text
            query = redact_text(query)

            results = self.provider.search(query, max_results=self.max_results)
            if not results:
                return None

            items = []
            for r in results:
                items.append(f"  <result>\n    <title>{r.title}</title>\n    <snippet>{r.snippet}</snippet>\n    <url>{r.url}</url>\n  </result>")
            results_xml = "\n".join(items)

            return (
                f"\n<web_search_results>\n"
                f"<search_query>{query}</search_query>\n"
                f"{results_xml}\n"
                f"</web_search_results>\n"
                f"<search_instructions>\n"
                f"Use the search results above to inform your answer. "
                f"Synthesise the information naturally in your own voice and personality. "
                f"Do not list URLs or cite sources — this is a voice assistant and citations are not useful when spoken aloud. "
                f"If the results are irrelevant or insufficient, answer from your own knowledge and do not mention that a search was performed.\n"
                f"</search_instructions>"
            )
        except Exception:
            return None
