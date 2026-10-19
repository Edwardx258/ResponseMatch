# Evaluation

`run_eval.py` compares the four matching modes on a labelled set and reports, per mode,
how many volunteers were recommended, how many of those were false positives, and recall.
The headline comparison is **dual_track vs llm_only** false-positive rate.

## Labelled set

`eval/labeled.jsonl` (git-ignored until we decide what can be published). One JSON object
per line:

```json
{"request": {"id": "q001", "text": "...", "lat": 47.6, "lon": -122.3},
 "acceptable": ["v03", "v07"]}
```

`acceptable` lists every volunteer a dispatcher would accept for the request. Label
against the same volunteer roster the run uses (`RESPONSEMATCH_VOLUNTEERS`).

## Running

```bash
uv run python eval/run_eval.py --labels eval/labeled.jsonl
```

The graph (and its Claude calls) runs once per request; the modes only re-merge the two
tracks, so cost scales with the number of requests, not modes.

Numbers from the heuristic backend or from `data/sample/` only show that the harness
works. Do not publish them as results.
