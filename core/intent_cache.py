"""
Semantic intent cache backed by ChromaDB.

Stores successful Claude-routed intents so future similar inputs
skip the LLM call entirely. Gets smarter over time as more intents
are cached from real usage.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class IntentCache:
    """
    Persistent semantic cache for intent routing results.

    Uses ChromaDB with sentence-transformer embeddings so similar
    phrasings (e.g. "de-stress the lights" ≈ "chill vibes mode")
    hit the same cached routing without needing exact token overlap.
    """

    def __init__(
        self,
        path: str = "~/.friday/intent_cache",
        collection_name: str = "intents",
        similarity_threshold: float = 0.75,
        ttl_days: float = 30,
    ):
        self.threshold = similarity_threshold
        self.ttl_days = ttl_days
        self._collection = None
        self._db_path = str(Path(path).expanduser())
        self._collection_name = collection_name

    def _get_collection(self):
        """Lazy-init ChromaDB so startup isn't blocked if the package is missing."""
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=self._db_path)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def query(self, text: str) -> Optional[Tuple[str, dict]]:
        """
        Return (workflow_name, entities) if a sufficiently similar intent is cached,
        otherwise None.
        """
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return None

            results = collection.query(
                query_texts=[text],
                n_results=1,
                include=["metadatas", "distances"],
            )

            if not results["ids"][0]:
                return None

            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            # For sentence embeddings in practice this stays in [0, 1]
            distance = results["distances"][0][0]
            similarity = 1.0 - distance

            if similarity < self.threshold:
                logger.debug(
                    "Cache miss (similarity=%.2f < threshold=%.2f) for: %s",
                    similarity,
                    self.threshold,
                    text,
                )
                return None

            metadata = results["metadatas"][0][0]
            doc_id = results["ids"][0][0]

            # TTL by created_at, not last_used: routing/entity decisions decay
            # with the codebase, and hits shouldn't immortalize an entry.
            if self.ttl_days and metadata.get("created_at"):
                try:
                    age = datetime.now() - datetime.fromisoformat(metadata["created_at"])
                    if age.total_seconds() > self.ttl_days * 86400:
                        collection.delete(ids=[doc_id])
                        logger.info("Intent cache: expired entry for %s (age %dd)",
                                    metadata.get("workflow_name"), age.days)
                        return None
                except ValueError:
                    pass

            # Increment hit count
            collection.update(
                ids=[doc_id],
                metadatas=[{
                    **metadata,
                    "hit_count": metadata.get("hit_count", 0) + 1,
                    "last_used": datetime.now().isoformat(),
                }],
            )

            logger.debug(
                "Cache hit (similarity=%.2f) → %s for: %s",
                similarity,
                metadata["workflow_name"],
                text,
            )
            return metadata["workflow_name"], json.loads(metadata["entities_json"])

        except Exception:
            logger.exception("Intent cache query failed, skipping cache")
            return None

    def store(self, text: str, workflow_name: str, entities: dict) -> None:
        """Persist a successful Claude routing result for future reuse."""
        try:
            collection = self._get_collection()
            doc_id = hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]

            collection.upsert(
                documents=[text],
                metadatas=[{
                    "workflow_name": workflow_name,
                    "entities_json": json.dumps(entities),
                    "hit_count": 0,
                    "created_at": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat(),
                }],
                ids=[doc_id],
            )
            logger.debug("Stored intent: '%s' → %s", text, workflow_name)

        except Exception:
            logger.exception("Intent cache store failed, skipping")

    def delete_workflow(self, workflow_name: str) -> int:
        """Drop every cached intent that routes to `workflow_name`.

        Maintenance hook (e.g. after a workflow is renamed or its entity schema
        changes). Returns the number of entries removed, 0 on any failure.
        """
        try:
            collection = self._get_collection()
            before = collection.count()
            if before == 0:
                return 0
            collection.delete(where={"workflow_name": workflow_name})
            removed = before - collection.count()
            if removed:
                logger.info("Intent cache: removed %d entr%s for %s",
                            removed, "y" if removed == 1 else "ies", workflow_name)
            return removed
        except Exception:
            logger.exception("Intent cache delete failed for %s", workflow_name)
            return 0

    def count(self) -> int:
        """Return total number of cached intents."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0
