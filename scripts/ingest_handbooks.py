"""Chunk handbook documents (.md, .txt, .pdf) and load them into the ChromaDB store.

    uv run python scripts/ingest_handbooks.py --folder data/handbooks
    uv run python scripts/ingest_handbooks.py --folder data/sample/handbook   # sample text
"""

from __future__ import annotations

import argparse
from pathlib import Path

from responsematch.config import Settings
from responsematch.retrieval import HandbookStore, make_embedder


def main() -> None:
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=settings.handbook_dir)
    ap.add_argument("--query", default="welfare check for an elderly resident", help="test query to run afterwards")
    args = ap.parse_args()

    store = HandbookStore(make_embedder(settings.embedder), path=settings.chroma_path)
    added = store.ingest_folder(args.folder)
    print(f"Upserted {added} chunks from {args.folder}; store now holds {len(store)} (embedder={settings.embedder})")
    for p in store.search(args.query, k=3):
        print(f"  {p.score:.3f}  {p.id}  {p.text[:80].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
