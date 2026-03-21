#!/usr/bin/env python3
"""
Seed the local Qdrant index with papers from the Semantic Scholar API.

Usage:
    # Seed with keyword search
    python scripts/seed_qdrant.py --keywords "transformer attention" --limit 500

    # Seed by topic / field of study
    python scripts/seed_qdrant.py --keywords "deep learning" --limit 1000 --year-from 2020

    # Show current index stats
    python scripts/seed_qdrant.py --stats

Requirements:
    uv run python scripts/seed_qdrant.py ...  (from repo root)
    or: activate .venv first, then run normally
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make sure the mabool package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "agents" / "mabool" / "api"))


async def main(args: argparse.Namespace) -> None:
    # Import here so we get nice error messages if qdrant-client is missing
    try:
        from mabool.agents.local_dense.qdrant_index import get_index_stats, index_documents
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Make sure qdrant-client[fastembed] is installed:")
        print("  uv add 'qdrant-client[fastembed]' --directory agents/mabool/api")
        sys.exit(1)

    if args.stats:
        stats = get_index_stats()
        print("── Qdrant Index Stats ──────────────────")
        for k, v in stats.items():
            print(f"  {k:25s}: {v}")
        return

    # Fetch papers via the Semantic Scholar public API
    import urllib.request
    import json

    keywords = args.keywords or "machine learning"
    limit = min(args.limit, 500)  # S2 max per request is 100; we batch
    year_from = args.year_from

    print(f"Fetching up to {limit} papers for: '{keywords}'")
    if year_from:
        print(f"  Year filter: >= {year_from}")

    base_url = "https://api.semanticscholar.org/graph/v1"
    fields = "corpusId,title,abstract,year,venue"
    params = f"query={urllib.parse.quote(keywords)}&fields={fields}&limit=100"

    papers: list[dict] = []
    offset = 0

    import urllib.parse

    while len(papers) < limit:
        batch_limit = min(100, limit - len(papers))
        url = f"{base_url}/paper/search?query={urllib.parse.quote(keywords)}&fields={fields}&limit={batch_limit}&offset={offset}"

        # Use S2 API key if available
        import os
        headers = {}
        api_key = os.environ.get("S2_API_KEY", "")
        if api_key:
            headers["x-api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  API error at offset {offset}: {e}")
            break

        batch = data.get("data", [])
        if not batch:
            break

        for p in batch:
            if not p.get("corpusId"):
                continue
            if year_from and (p.get("year") or 0) < year_from:
                continue
            papers.append({
                "corpus_id": str(p["corpusId"]),
                "title": p.get("title") or "",
                "abstract": p.get("abstract") or "",
                "year": p.get("year"),
                "venue": p.get("venue") or "",
            })

        offset += len(batch)
        print(f"  Fetched {len(papers)} papers so far…", end="\r")

        if len(batch) < 100:
            break  # No more results

    print(f"\nFetched {len(papers)} papers total.")

    if not papers:
        print("Nothing to index.")
        return

    print("Embedding and indexing (this may take a moment on first run)…")
    added = await index_documents(papers)
    print(f"Done! Added {added} papers to the local Qdrant index.")
    stats = get_index_stats()
    print(f"Total papers in index: {stats.get('vectors_count', '?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the local Qdrant index.")
    parser.add_argument("--keywords", type=str, default="", help="Search keywords for S2")
    parser.add_argument("--limit", type=int, default=200, help="Max papers to fetch (default 200)")
    parser.add_argument("--year-from", type=int, default=None, help="Only include papers >= this year")
    parser.add_argument("--stats", action="store_true", help="Show current index stats and exit")
    args = parser.parse_args()
    asyncio.run(main(args))
