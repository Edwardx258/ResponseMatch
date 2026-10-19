"""The LangGraph workflow.

    START → parse → candidates ┬→ rule_track ───────────────┬→ combine → END
                               └→ retrieve → semantic_track ┘

The two tracks run in parallel branches; combine waits for both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .combine import combine
from .config import Mode, Settings
from .llm import StructuredLLM
from .parsing import HeuristicParser, LLMParser, Parser
from .retrieval import HandbookStore, make_embedder, query_for
from .rules import haversine_km, score_all
from .schemas import (
    GeoPoint,
    Judgement,
    Match,
    MatchResponse,
    ParsedRequest,
    Passage,
    RuleScore,
    ServiceRequest,
    Volunteer,
)
from .semantic import Judge, KeywordJudge, LLMJudge


class MatchState(TypedDict, total=False):
    request: ServiceRequest
    mode: Mode
    parsed: ParsedRequest
    candidates: list[Volunteer]
    passages: list[Passage]
    rule_scores: dict[str, RuleScore]
    judgements: dict[str, Judgement]
    matches: list[Match]


@dataclass
class Deps:
    settings: Settings
    parser: Parser
    judge: Judge
    store: HandbookStore
    volunteers: list[Volunteer]


def load_volunteers(path: Path) -> list[Volunteer]:
    return [Volunteer.model_validate(v) for v in json.loads(path.read_text(encoding="utf-8"))]


def make_llm(settings: Settings) -> StructuredLLM:
    if settings.backend == "deepseek":
        from .deepseek import DEFAULT_MODEL, StructuredDeepSeek

        return StructuredDeepSeek(settings.model or DEFAULT_MODEL, base_url=settings.deepseek_base_url)
    if settings.backend == "claude":
        from .claude import DEFAULT_MODEL, StructuredClaude

        return StructuredClaude(settings.model or DEFAULT_MODEL)
    raise ValueError(f"unknown LLM backend: {settings.backend}")


def build_deps(settings: Settings | None = None) -> Deps:
    settings = settings or Settings()
    if settings.backend == "heuristic":
        parser, judge = HeuristicParser(), KeywordJudge()
    else:
        llm = make_llm(settings)
        parser, judge = LLMParser(llm), LLMJudge(llm)
    store = HandbookStore(make_embedder(settings.embedder), path=settings.chroma_path)
    if len(store) == 0 and settings.handbook_dir.exists():
        store.ingest_folder(settings.handbook_dir)
    return Deps(settings, parser, judge, store, load_volunteers(settings.volunteers_path))


def _nearest(request: ServiceRequest, volunteers: list[Volunteer], n: int) -> list[Volunteer]:
    """The judge sees at most n candidates; pick the closest (all of them if no location)."""
    loc = request.location
    if loc is None:
        return volunteers[:n]
    return sorted(volunteers, key=lambda v: haversine_km(loc, GeoPoint(lat=v.lat, lon=v.lon)))[:n]


def build_graph(deps: Deps):
    s = deps.settings

    def parse(state: MatchState) -> MatchState:
        return {"parsed": deps.parser.parse(state["request"])}

    def candidates(state: MatchState) -> MatchState:
        return {"candidates": _nearest(state["request"], deps.volunteers, s.max_candidates)}

    def rule_track(state: MatchState) -> MatchState:
        return {"rule_scores": score_all(state["parsed"], state["request"].location, state["candidates"])}

    def retrieve(state: MatchState) -> MatchState:
        return {"passages": deps.store.search(query_for(state["parsed"]), k=s.top_k_passages)}

    def semantic_track(state: MatchState) -> MatchState:
        return {
            "judgements": deps.judge.judge(
                state["request"], state["parsed"], state["candidates"], state["passages"]
            )
        }

    def merge(state: MatchState) -> MatchState:
        mode = state.get("mode", s.mode)
        return {"matches": combine(state["candidates"], state["rule_scores"], state["judgements"], s, mode)}

    g = StateGraph(MatchState)
    for name, fn in [
        ("parse", parse),
        ("candidates", candidates),
        ("rule_track", rule_track),
        ("retrieve", retrieve),
        ("semantic_track", semantic_track),
        ("combine", merge),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    g.add_edge("parse", "candidates")
    g.add_edge("candidates", "rule_track")
    g.add_edge("candidates", "retrieve")
    g.add_edge("retrieve", "semantic_track")
    g.add_edge(["rule_track", "semantic_track"], "combine")
    g.add_edge("combine", END)
    return g.compile()


def run_match(graph, request: ServiceRequest, mode: Mode) -> MatchResponse:
    state = graph.invoke({"request": request, "mode": mode})
    return MatchResponse(
        request_id=request.id,
        mode=mode,
        parsed=state["parsed"],
        passages=state["passages"],
        matches=state["matches"],
    )
