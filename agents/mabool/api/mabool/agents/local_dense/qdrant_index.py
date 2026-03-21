"""
Local Qdrant dense-retrieval index.

Replaces the internal Vespa DenseAgent for local deployments.
Uses fastembed (via qdrant-client[fastembed]) — no external service needed.

Collection grows automatically:  every time a search completes, newly found
papers are embedded and upserted into the local Qdrant store.  Subsequent
searches benefit from the accumulated corpus.

Setup (one-time, optional seeding):
    python scripts/seed_qdrant.py --keywords "transformer attention" --limit 500

No setup is required: the first regular search will already start building
the index in the background.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

COLLECTION_NAME = "asta_papers"
# Small but high-quality model: 33M params, 384-dim, fast on CPU
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Qdrant data stored next to the search_history.db
_QDRANT_PATH = Path(__file__).parent.parent.parent.parent / "qdrant_data"


def _get_client():
    """Return a local QdrantClient, or None if qdrant-client is not installed."""
    try:
        from qdrant_client import QdrantClient  # noqa: PLC0415

        _QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(_QDRANT_PATH))
    except ImportError:
        logger.debug("qdrant-client not installed; local dense search disabled")
        return None
    except Exception as exc:
        logger.warning("Could not open Qdrant store: %s", exc)
        return None


# ── Search ────────────────────────────────────────────────────────────────────

def _sync_search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Synchronous search — run inside a thread via asyncio.to_thread."""
    client = _get_client()
    if client is None:
        return []

    try:
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=top_k,
            with_payload=True,
        )
        return [(str(r.metadata.get("corpus_id", "")), r.score) for r in results if r.metadata.get("corpus_id")]
    except Exception as exc:
        logger.debug("Qdrant search failed (collection may be empty): %s", exc)
        return []


async def search_local_qdrant(query: str, top_k: int = 60) -> list[tuple[str, float]]:
    """Return (corpus_id, score) pairs from the local Qdrant index."""
    return await asyncio.to_thread(_sync_search, query, top_k)


# ── Indexing ──────────────────────────────────────────────────────────────────

def _sync_index(texts: list[str], ids: list[str], metadatas: list[dict]) -> int:
    """Upsert documents into the local Qdrant store. Returns number added."""
    client = _get_client()
    if client is None:
        return 0

    try:
        client.add(
            collection_name=COLLECTION_NAME,
            documents=texts,
            ids=ids,
            metadata=metadatas,
        )
        return len(texts)
    except Exception as exc:
        logger.warning("Qdrant indexing error: %s", exc)
        return 0


async def index_documents(documents: list[dict]) -> int:
    """
    Embed and upsert a batch of paper dicts into Qdrant.

    Each dict should have at least:
        corpus_id: str | int
        title:     str | None
        abstract:  str | None
    """
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for doc in documents:
        cid = str(doc.get("corpus_id", "")).strip()
        if not cid:
            continue
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or "").strip()
        text = f"{title}. {abstract}".strip(" .")
        if not text:
            continue
        texts.append(text)
        ids.append(cid)
        metadatas.append({"corpus_id": cid, "year": doc.get("year"), "venue": doc.get("venue") or ""})

    if not texts:
        return 0

    added = await asyncio.to_thread(_sync_index, texts, ids, metadatas)
    if added:
        logger.info("Qdrant: indexed %d papers", added)
    return added


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_index_stats() -> dict:
    """Return basic stats about the local Qdrant index (synchronous)."""
    client = _get_client()
    if client is None:
        return {"available": False, "reason": "qdrant-client not installed"}

    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "available": True,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "collection": COLLECTION_NAME,
            "embedding_model": _EMBEDDING_MODEL,
        }
    except Exception:
        return {"available": True, "vectors_count": 0, "points_count": 0,
                "collection": COLLECTION_NAME, "status": "empty (no papers indexed yet)"}
