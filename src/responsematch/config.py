"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["claude", "deepseek", "heuristic"]
Mode = Literal["dual_track", "weighted", "llm_only", "rule_only"]
MODES: tuple[Mode, ...] = ("dual_track", "weighted", "llm_only", "rule_only")

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # which model provider runs the parser and the judge; "heuristic" = offline keyword baselines
    backend: Backend = field(
        default_factory=lambda: _env("RESPONSEMATCH_BACKEND", "claude")  # type: ignore[arg-type]
    )
    # empty = the backend's default model (see claude.py / deepseek.py)
    model: str = field(default_factory=lambda: _env("RESPONSEMATCH_MODEL", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )

    volunteers_path: Path = field(
        default_factory=lambda: Path(_env("RESPONSEMATCH_VOLUNTEERS", str(ROOT / "data/sample/volunteers.json")))
    )
    handbook_dir: Path = field(
        default_factory=lambda: Path(_env("RESPONSEMATCH_HANDBOOKS", str(ROOT / "data/sample/handbook")))
    )
    chroma_path: Path = field(default_factory=lambda: Path(_env("RESPONSEMATCH_CHROMA", str(ROOT / ".chroma"))))
    # "hash" works offline; "default" uses Chroma's MiniLM model (downloads on first use)
    embedder: Literal["hash", "default"] = field(
        default_factory=lambda: _env("RESPONSEMATCH_EMBEDDER", "hash")  # type: ignore[arg-type]
    )

    mode: Mode = field(default_factory=lambda: _env("RESPONSEMATCH_MODE", "dual_track"))  # type: ignore[arg-type]
    rule_weight: float = field(default_factory=lambda: float(_env("RESPONSEMATCH_RULE_WEIGHT", "0.5")))
    rule_min: float = field(default_factory=lambda: float(_env("RESPONSEMATCH_RULE_MIN", "0.35")))
    llm_min: float = field(default_factory=lambda: float(_env("RESPONSEMATCH_LLM_MIN", "0.5")))
    weighted_min: float = field(default_factory=lambda: float(_env("RESPONSEMATCH_WEIGHTED_MIN", "0.5")))

    top_k_passages: int = field(default_factory=lambda: int(_env("RESPONSEMATCH_TOP_K", "4")))
    max_candidates: int = field(default_factory=lambda: int(_env("RESPONSEMATCH_MAX_CANDIDATES", "8")))
