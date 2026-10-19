from pathlib import Path

import pytest

from responsematch.config import Settings
from responsematch.graph import Deps, load_volunteers
from responsematch.parsing import HeuristicParser
from responsematch.retrieval import HandbookStore, HashingEmbedder
from responsematch.schemas import ServiceRequest
from responsematch.semantic import KeywordJudge

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def settings() -> Settings:
    return Settings(backend="heuristic", embedder="hash", volunteers_path=SAMPLE / "volunteers.json",
                    handbook_dir=SAMPLE / "handbook")


@pytest.fixture
def store() -> HandbookStore:
    s = HandbookStore(HashingEmbedder(), path=None, collection=f"test-{id(object())}")
    s.ingest_folder(SAMPLE / "handbook")
    return s


@pytest.fixture
def deps(settings, store) -> Deps:
    return Deps(settings, HeuristicParser(), KeywordJudge(), store, load_volunteers(settings.volunteers_path))


@pytest.fixture
def sample_requests() -> dict[str, ServiceRequest]:
    import json

    rows = [json.loads(line) for line in (SAMPLE / "requests.jsonl").read_text().splitlines() if line.strip()]
    return {r["id"]: ServiceRequest.model_validate(r) for r in rows}
