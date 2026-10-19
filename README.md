# ResponseMatch

> **Status: work in progress.** This is a tech-stack spike. Nothing here has been
> evaluated yet, and no results should be quoted from it.

Match community service requests (Seattle 311 style) to volunteers with a LangGraph
workflow that scores every candidate on two independent tracks and only recommends
volunteers both tracks agree on.

```
START → parse → candidates ┬→ rule_track ───────────────┬→ combine → END
                           └→ retrieve → semantic_track ┘
```

| Stage | What it does | Code |
|---|---|---|
| parse | The LLM extracts category, urgency, required skills, certifications and languages | `parsing.py` |
| candidates | Nearest volunteers, capped at `RESPONSEMATCH_MAX_CANDIDATES` | `graph.py` |
| rule_track | Hard constraints (availability, range, certifications, language) and a skill/proximity score | `rules.py` |
| retrieve | Handbook passages from ChromaDB | `retrieval.py` |
| semantic_track | The LLM judges each candidate, grounded in the passages, and cites them | `semantic.py` |
| combine | `dual_track`, `weighted`, `llm_only` or `rule_only` | `combine.py` |

Stack: Python · LangGraph · ChromaDB · FastAPI, with a pluggable LLM backend.

## LLM backends

| `RESPONSEMATCH_BACKEND` | Default model | Structured output | Key |
|---|---|---|---|
| `claude` | `claude-opus-5-5` | Schema-constrained structured outputs, server-side fallback on refusals | `ANTHROPIC_API_KEY` |
| `deepseek` | `deepseek-chat` | JSON mode; the schema is in the prompt and replies are validated, with one retry | `DEEPSEEK_API_KEY` |
| `heuristic` | — | Keyword rules, offline; for tests and development only | none |

Set `RESPONSEMATCH_MODEL` to override the model. Both LLM backends implement the same
`StructuredLLM` interface (`llm.py`), so running `eval/run_eval.py` once per backend
compares them on the same labelled set.

## Quick start

```bash
uv sync
uv run pytest                                            # offline, no API key needed

# offline demo with keyword baselines instead of Claude
RESPONSEMATCH_BACKEND=heuristic uv run python scripts/demo.py

# with Claude: set ANTHROPIC_API_KEY (or `ant auth login`), then
uv run python scripts/demo.py

# with DeepSeek
DEEPSEEK_API_KEY=... RESPONSEMATCH_BACKEND=deepseek uv run python scripts/demo.py
uv run uvicorn responsematch.api:app --reload            # POST /match, GET /health
```

```bash
curl -X POST localhost:8000/match -H 'content-type: application/json' \
  -d '{"text": "Someone fell and is injured, need first aid", "lat": 47.565, "lon": -122.386}'
```

Configuration is in `.env.example`.

## Data

`data/sample/` is **synthetic** (hand-written volunteers, requests and placeholder
handbook text) so the pipeline can run and be tested. Real data comes from public sources:

- **Requests** — Seattle's public Customer Service Requests dataset on data.seattle.gov:
  `scripts/fetch_seattle_311.py --dataset-id <id>` (it gives request types, dates and
  locations; request descriptions are not included, so free text still has to be written
  or sampled separately).
- **Handbooks** — public preparedness and volunteer guides listed in `data/sources.txt`
  (see `data/sources.txt.example`), downloaded with `scripts/fetch_documents.py` and
  loaded with `scripts/ingest_handbooks.py --folder data/handbooks`.
- **Volunteers** — always synthetic; no real people's data.

Check each source's terms of use before redistributing anything.

## Evaluation

See `eval/README.md`: a labelled set, and `eval/run_eval.py` to compare the dual track
with the LLM-only baseline on false positives. RAGAS faithfulness for the semantic track
is the next step.

## Roadmap

- [x] Workflow skeleton, both tracks, API, offline tests
- [ ] Ingest real public handbooks; switch to a real embedding model
- [ ] Build a labelled evaluation set
- [ ] Measure dual_track vs llm_only; add RAGAS faithfulness
- [ ] Compare Claude and DeepSeek on the same labelled set (quality, cost, latency)
- [ ] Map Seattle 311 request types to categories
