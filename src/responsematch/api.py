"""FastAPI service: POST /match runs the workflow for one request.

Run with:  uv run uvicorn responsematch.api:app --reload
"""

from __future__ import annotations

import uuid
from functools import lru_cache

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .claude import ClaudeError, ClaudeRefusal
from .config import MODES, Mode
from .graph import Deps, build_deps, build_graph, run_match
from .schemas import MatchResponse, ServiceRequest


class MatchRequest(BaseModel):
    text: str
    lat: float | None = None
    lon: float | None = None
    category_hint: str | None = None
    id: str | None = None
    mode: Mode | None = None


def create_app(deps: Deps | None = None) -> FastAPI:
    app = FastAPI(title="ResponseMatch", version="0.1.0")

    @lru_cache(maxsize=1)
    def workflow():
        d = deps or build_deps()
        return d, build_graph(d)

    @app.get("/health")
    def health():
        d, _ = workflow()
        return {
            "status": "ok",
            "backend": d.settings.backend,
            "volunteers": len(d.volunteers),
            "handbook_chunks": len(d.store),
            "modes": list(MODES),
        }

    @app.post("/match", response_model=MatchResponse)
    def match(body: MatchRequest) -> MatchResponse:
        d, graph = workflow()
        request = ServiceRequest(
            id=body.id or uuid.uuid4().hex[:8],
            text=body.text,
            lat=body.lat,
            lon=body.lon,
            category_hint=body.category_hint,
        )
        try:
            return run_match(graph, request, body.mode or d.settings.mode)
        except ClaudeRefusal as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except anthropic.RateLimitError as e:
            raise HTTPException(status_code=503, detail="Claude API rate limited; retry shortly") from e
        except (ClaudeError, anthropic.APIError) as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e

    return app


app = create_app()
