"""Arm A — reflection memory: generative-agents-style observations and
reflections distilled nightly from conversations, retrieved per-query at eval
time as a <learned_memory> block.

Write path (nightly): the local model extracts durable observations (with
importance 1-10) from the day's exchanges; once enough new observations
accumulate, a second pass synthesizes higher-level reflections. Both land in
research.db `memories` (append-only; retirement is a flag).

Read path (eval/replay only): top-k memories scored by
alpha*similarity + beta*recency + gamma*importance. The vector index is
injectable; without one, similarity falls back to 0 and recency+importance
carry the ranking (still useful, trivially testable).

The artifact for versioning is a dated snapshot of the memories table plus the
pinned retrieval weights; revert = repoint `current` (retrieval then filters to
memories that existed in that snapshot).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from research import artifacts, provenance
from research.db import ResearchStore

logger = logging.getLogger(__name__)

ARM = "memory"
OBSERVE_MODEL = "qwen3:8b"
REFLECT_EVERY = 20  # new observations between reflection passes

# Pinned retrieval weights (recorded in every snapshot's config.json).
WEIGHTS = {"alpha_similarity": 0.6, "beta_recency": 0.2, "gamma_importance": 0.2,
           "tau_days": 14.0, "k": 6}

_OBSERVE_PROMPT = """Extract durable facts about the user from this conversation transcript: preferences, routines, corrections, plans, relationships to things (not people's private details). Skip pleasantries and one-off context.

Transcript (FEEDBACK lines mark exchanges the user liked (+1) or disliked (-1)):
{transcript}

Answer with only a JSON array, each item {{"text": "<one-sentence fact>", "importance": <1-10, how much this should influence future replies>}}. Empty array if nothing durable."""

_REFLECT_PROMPT = """These are recent observations about one user, newest last:
{observations}

What are up to 3 higher-level patterns or preferences they imply that no single observation states? Answer with only a JSON array of {{"text": "<one-sentence insight>", "importance": <1-10>}}. Empty array if none."""


class VectorIndex(Protocol):
    def add(self, memory_id: int, text: str) -> None: ...
    def query(self, text: str, k: int) -> list[tuple[int, float]]: ...  # (id, similarity)


class ChromaIndex:
    """Persistent ChromaDB index (same embedding stack as the intent cache)."""

    def __init__(self, path: Path):
        import chromadb
        client = chromadb.PersistentClient(path=str(path))
        self._col = client.get_or_create_collection("memories")

    def add(self, memory_id: int, text: str) -> None:
        self._col.upsert(ids=[str(memory_id)], documents=[text])

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_texts=[text], n_results=min(k, self._col.count()))
        ids = res["ids"][0]
        # Chroma returns L2 distances for the default space; map to a bounded
        # similarity in (0, 1] — monotonic is all the scorer needs.
        dists = res["distances"][0]
        return [(int(i), 1.0 / (1.0 + d)) for i, d in zip(ids, dists)]


def _default_llm(prompt: str, *, base_url: str = "http://localhost:11434") -> str:
    import requests

    from research.generate import strip_think
    from research.shadow import supports_thinking

    payload = {"model": OBSERVE_MODEL, "stream": False,
               "messages": [{"role": "user", "content": prompt}]}
    if supports_thinking(OBSERVE_MODEL):
        payload["think"] = False
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
    return strip_think((resp.json().get("message") or {}).get("content", ""))


def _parse_items(raw: str) -> list[dict]:
    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        items = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        logger.warning("memory agent: unparseable model output: %r", raw[:120])
        return []
    out = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            importance = float(item.get("importance", 5))
        except (TypeError, ValueError):
            importance = 5.0
        out.append({"text": text, "importance": max(1.0, min(10.0, importance))})
    return out


class MemoryAgent:
    def __init__(
        self,
        store: ResearchStore,
        *,
        artifacts_dir: Path = artifacts.DEFAULT_ARTIFACTS_DIR,
        index: Optional[VectorIndex] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.store = store
        self.artifacts_dir = Path(artifacts_dir).expanduser()
        self._index = index
        self._llm = llm_fn or _default_llm

    # ------------------------------------------------------------ write path
    def observe(self, since_ts: float) -> int:
        """Extract observations from chat exchanges since the cutoff."""
        from research.approaches.prompt_evolver import format_conversations

        transcript = format_conversations(self.store, since_ts)
        if not transcript:
            return 0
        items = _parse_items(self._llm(_OBSERVE_PROMPT.format(transcript=transcript)))
        exchange_ids = [e["id"] for e in self.store.exchanges_since(since_ts)
                        if e["route"] == "chat"]
        # Which turns this arm read, whether or not they yielded a memory.
        self.store.emit_all("memory.consumed", subject_type="exchange",
                            subject_ids=exchange_ids, arm=ARM,
                            detail={"observations_extracted": len(items)})
        return self._insert(items, kind="observation", sources=exchange_ids)

    def maybe_reflect(self) -> int:
        """Synthesize reflections once REFLECT_EVERY observations accumulated
        since the last reflection."""
        last_reflection_ts = self.store.query(
            "SELECT COALESCE(MAX(ts), 0) AS ts FROM memories WHERE kind = 'reflection'"
        )[0]["ts"]
        rows = self.store.query(
            "SELECT id, text FROM memories WHERE kind = 'observation'"
            " AND retired = 0 AND ts > ? ORDER BY ts",
            (last_reflection_ts,),
        )
        if len(rows) < REFLECT_EVERY:
            return 0
        listing = "\n".join(f"- {r['text']}" for r in rows)
        items = _parse_items(self._llm(_REFLECT_PROMPT.format(observations=listing)))
        return self._insert(items[:3], kind="reflection",
                            sources=[r["id"] for r in rows])

    def _insert(self, items: list[dict], *, kind: str, sources: list[int]) -> int:
        inserted = 0
        for item in items:
            memory_id = self.store.execute(
                "INSERT INTO memories (ts, kind, text, importance, source_exchange_ids)"
                " VALUES (?, ?, ?, ?, ?)",
                (time.time(), kind, item["text"], item["importance"],
                 json.dumps(sources)),
            )
            self.store.emit(
                "memory.observed" if kind == "observation" else "memory.reflected",
                subject_type="memory", subject_id=memory_id, arm=ARM,
                detail={"importance": item["importance"],
                        "chars": len(item["text"]), "sources": sources},
            )
            if self._index is not None:
                try:
                    self._index.add(memory_id, item["text"])
                except Exception:
                    logger.warning("memory index add failed", exc_info=True)
            inserted += 1
        return inserted

    # ------------------------------------------------------------- read path
    def retrieve(self, query: str, k: int = WEIGHTS["k"],
                 *, max_id: Optional[int] = None) -> list[dict]:
        """Top-k live memories for a query. `max_id` pins retrieval to a
        snapshot (memories that existed when that version was taken)."""
        where = "retired = 0" + (" AND id <= ?" if max_id is not None else "")
        params = (max_id,) if max_id is not None else ()
        rows = self.store.query(
            f"SELECT id, ts, kind, text, importance FROM memories WHERE {where}",
            params,
        )
        if not rows:
            return []

        similarity: dict[int, float] = {}
        if self._index is not None:
            try:
                similarity = dict(self._index.query(query, k=max(k * 3, 20)))
            except Exception:
                logger.warning("memory index query failed", exc_info=True)

        now = time.time()
        w = WEIGHTS

        def score(row: dict) -> float:
            age_days = max(0.0, (now - row["ts"]) / 86400)
            return (w["alpha_similarity"] * similarity.get(row["id"], 0.0)
                    + w["beta_recency"] * math.exp(-age_days / w["tau_days"])
                    + w["gamma_importance"] * row["importance"] / 10.0)

        return sorted(rows, key=score, reverse=True)[:k]

    def system_block_for(self, query: str, *, max_id: Optional[int] = None) -> str:
        memories = self.retrieve(query, max_id=max_id)
        if not memories:
            return ""
        lines = "\n".join(f"- {m['text']}" for m in memories)
        return f"<learned_memory>\n{lines}\n</learned_memory>"

    # ----------------------------------------------------------- versioning
    def snapshot(self, date_str: str) -> str:
        """Dated snapshot of the memories table + pinned weights; advances
        `current`. Returns the version name."""
        rows = self.store.query("SELECT * FROM memories ORDER BY id")
        version_dir = artifacts.new_version(ARM, date_str, self.artifacts_dir)
        max_memory_id = rows[-1]["id"] if rows else 0
        (version_dir / "memories.json").write_text(json.dumps(rows, indent=1))
        (version_dir / "config.json").write_text(json.dumps(
            {"weights": WEIGHTS, "observe_model": OBSERVE_MODEL,
             "max_memory_id": max_memory_id}, indent=1))

        source_exchanges = sorted({
            eid for r in rows
            for eid in json.loads(r["source_exchange_ids"] or "[]")
        })
        provenance.write_manifest(
            version_dir, ARM,
            built_ts=time.time(),
            git_rev=provenance.git_rev(),
            inputs={
                "memories_total": len(rows),
                "observations": sum(1 for r in rows if r["kind"] == "observation"),
                "reflections": sum(1 for r in rows if r["kind"] == "reflection"),
                "retired": sum(1 for r in rows if r["retired"]),
                "source_exchanges": source_exchanges,
                "max_memory_id": max_memory_id,
            },
            params={"weights": WEIGHTS, "observe_model": OBSERVE_MODEL},
        )
        version = f"{ARM}/{version_dir.name}"
        self.store.emit("artifact.created", subject_type="artifact",
                        subject_id=version, arm=ARM, artifact_version=version,
                        detail={"memories": len(rows),
                                "max_memory_id": max_memory_id})
        artifacts.advance_current(ARM, version_dir.name, self.artifacts_dir)
        self.store.emit("artifact.advanced", subject_type="artifact",
                        subject_id=version, arm=ARM, artifact_version=version)
        return version_dir.name

    def current_max_id(self) -> Optional[int]:
        """max_id pin from the current snapshot (None = no snapshot yet)."""
        path = artifacts.current_path(ARM, self.artifacts_dir)
        if path is None:
            return None
        cfg = path / "config.json"
        if not cfg.exists():
            return None
        return int(json.loads(cfg.read_text()).get("max_memory_id", 0)) or None
