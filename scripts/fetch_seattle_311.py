"""Download public Seattle customer service request (311) records via the Socrata API.

Find the dataset on https://data.seattle.gov (search "Customer Service Requests"),
copy its 4x4 id from the API endpoint (".../resource/<id>.json"), and check the
dataset's terms of use before redistributing anything you download.

    uv run python scripts/fetch_seattle_311.py --dataset-id abcd-1234 --limit 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

BASE = "https://data.seattle.gov/resource/{dataset}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", required=True, help="Socrata 4x4 id, e.g. abcd-1234")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--where", default=None, help="optional SoQL $where filter")
    ap.add_argument("--out", type=Path, default=Path("data/raw/seattle_311.json"))
    args = ap.parse_args()

    params: dict[str, str | int] = {"$limit": args.limit}
    if args.where:
        params["$where"] = args.where
    resp = httpx.get(BASE.format(dataset=args.dataset_id), params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    columns = sorted({k for row in rows for k in row})
    print(f"Saved {len(rows)} rows to {args.out}")
    print("Columns:", ", ".join(columns))


if __name__ == "__main__":
    main()
