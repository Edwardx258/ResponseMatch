"""Data models shared by every stage of the pipeline.

Models that Claude fills in through structured outputs (ParsedRequest,
Judgements) avoid numeric bounds in the schema; scores are clamped in code.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


URGENCY_LEVEL = {Urgency.low: 0, Urgency.medium: 1, Urgency.high: 2, Urgency.critical: 3}


class GeoPoint(BaseModel):
    lat: float
    lon: float


# ── inputs ──
class ServiceRequest(BaseModel):
    id: str
    text: str
    lat: float | None = None
    lon: float | None = None
    category_hint: str | None = None  # e.g. the 311 request type, when known

    @property
    def location(self) -> GeoPoint | None:
        if self.lat is None or self.lon is None:
            return None
        return GeoPoint(lat=self.lat, lon=self.lon)


class Volunteer(BaseModel):
    id: str
    name: str
    skills: list[str]
    languages: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    lat: float
    lon: float
    max_distance_km: float = 10.0
    available: bool = True


# ── stage 1: parsing (filled by Claude or the heuristic parser) ──
class ParsedRequest(BaseModel):
    category: str = Field(description="Short request category, e.g. 'welfare check' or 'debris removal'.")
    urgency: Urgency
    keywords: list[str] = Field(description="Lower-case keywords that describe the need.")
    required_skills: list[str] = Field(description="Volunteer skills the request needs, lower-case.")
    required_certifications: list[str] = Field(
        default_factory=list, description="Certifications that are mandatory, e.g. 'first aid'."
    )
    languages: list[str] = Field(default_factory=list, description="Languages the requester needs, if stated.")
    location_text: str | None = Field(default=None, description="Location as written in the request, if any.")
    summary: str = Field(description="One-sentence summary of the need.")


# ── stage 2: retrieval ──
class Passage(BaseModel):
    id: str
    source: str
    text: str
    score: float


# ── stage 3a: rule track ──
class RuleScore(BaseModel):
    volunteer_id: str
    score: float
    eligible: bool
    distance_km: float | None
    reasons: list[str]


# ── stage 3b: semantic track (filled by Claude or the keyword judge) ──
class Judgement(BaseModel):
    volunteer_id: str
    score: float = Field(description="Fit between 0 and 1.")
    rationale: str = Field(description="One sentence on why this volunteer does or does not fit.")
    cited_passages: list[str] = Field(default_factory=list, description="Ids of handbook passages relied on.")


class Judgements(BaseModel):
    judgements: list[Judgement]


# ── output ──
class Match(BaseModel):
    volunteer_id: str
    name: str
    recommended: bool
    combined_score: float
    rule_score: float | None
    llm_score: float | None
    distance_km: float | None
    reasons: list[str]
    rationale: str | None = None
    cited_passages: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    request_id: str
    mode: str
    parsed: ParsedRequest
    passages: list[Passage]
    matches: list[Match]
