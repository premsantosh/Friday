import time
from typing import Any, Optional
from collections import OrderedDict
import threading

class TTLCache:

    def __init__(self, max_size: int = 500, ttl_sec: int = 3600):
        self._max_size = max_size
        self._ttl_sec = ttl_sec
        self.store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def insert(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            expires_at = time.time() + (ttl or self._ttl_sec)
            self.store[key] = (value, expires_at)
            self.store.move_to_end(key)
            if len(self.store) > self._max_size:
                self.store.popitem(last=False)

    def evict(self, key: str):
        with self._lock:
            self.store.pop(key, None)


    def fetch(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.store:
                return None
            value, expires_at = self.store[key]

            if time.time() > expires_at:
                del self.store[key]
                return None
            self.store.move_to_end(key)
            return value


class FridayCache:
    """High-level cache for Friday's hot data."""

    def __init__(self):
        # Short TTL for volatile state
        self.device_state = TTLCache(max_size=100, ttl_sec=300)      # 5 min
        # Medium TTL for conversation context
        self.conversation = TTLCache(max_size=50, ttl_sec=1800)      # 30 min
        # Longer TTL for resolved entities
        self.entities = TTLCache(max_size=200, ttl_sec=7200)         # 2 hrs
        # Response cache - skip Claude entirely for repeated queries
        self.response_cache = TTLCache(max_size=100, ttl_sec=600)    # 10 min
        # Search results cache - avoid re-searching identical queries
        self.search_results = TTLCache(max_size=50, ttl_sec=900)     # 15 min

    def cache_response(self, query_fingerprint: str, response: str):
        """Cache a Claude response to serve identical/similar queries instantly."""
        self.response_cache.insert(query_fingerprint, response)

    def get_cached_response(self, query_fingerprint: str) -> Optional[str]:
        """Check if we can skip Claude entirely."""
        return self.response_cache.fetch(query_fingerprint)

    def cache_search(self, fingerprint: str, results: str):
        """Cache a search context block."""
        self.search_results.insert(fingerprint, results)

    def get_cached_search(self, fingerprint: str) -> Optional[str]:
        """Return cached search context or None."""
        return self.search_results.fetch(fingerprint)

