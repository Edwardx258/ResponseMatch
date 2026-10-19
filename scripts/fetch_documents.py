"""Download public handbook documents listed in data/sources.txt into data/handbooks/.

data/sources.txt has one URL per line (lines starting with # are ignored). Only list
documents whose licence allows this use; U.S. federal government works (for example
FEMA and Ready.gov publications) are generally in the public domain, while many
non-profit materials are copyrighted.

    uv run python scripts/fetch_documents.py
    uv run python scripts/ingest_handbooks.py --folder data/handbooks
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx


def filename_for(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    name = path.rsplit("/", 1)[-1] or urlparse(url).netloc
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=Path("data/sources.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/handbooks"))
    args = ap.parse_args()

    urls = [
        line.strip()
        for line in args.sources.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    for url in urls:
        resp = httpx.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        name = filename_for(url)
        if "pdf" in resp.headers.get("content-type", "") and not name.lower().endswith(".pdf"):
            name += ".pdf"
        (args.out / name).write_bytes(resp.content)
        print(f"{url} -> {args.out / name} ({len(resp.content) // 1024} KB)")


if __name__ == "__main__":
    main()
