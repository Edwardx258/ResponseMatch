"""Stage 1: turn a free-text request into a ParsedRequest."""

from __future__ import annotations

import re
from typing import Protocol

from .llm import StructuredLLM
from .schemas import ParsedRequest, ServiceRequest, Urgency

PARSER_SYSTEM = """You triage community service requests (Seattle 311 style) for a volunteer \
dispatch system. Extract what a dispatcher needs to find the right volunteer.

- urgency: critical = immediate risk to life or safety; high = vulnerable person or \
worsening situation needing help today; medium = needs help within days; low = routine.
- required_skills / required_certifications / languages: only what the request actually \
implies. Use short lower-case phrases such as "first aid", "heavy lifting", "spanish".
- Do not invent details that are not in the request."""


class Parser(Protocol):
    def parse(self, request: ServiceRequest) -> ParsedRequest: ...


class LLMParser:
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def parse(self, request: ServiceRequest) -> ParsedRequest:
        hint = f"\nReported category: {request.category_hint}" if request.category_hint else ""
        prompt = f"<request>\n{request.text}\n</request>{hint}"
        return self.llm.call(system=PARSER_SYSTEM, prompt=prompt, schema=ParsedRequest, effort="low")


# ── offline baseline: keyword rules, used for tests and development without an API key ──
_URGENCY = [
    (Urgency.critical, ("trapped", "unconscious", "not breathing", "fire", "flooding into", "gas leak")),
    (Urgency.high, ("elderly", "no heat", "no power", "injured", "medication", "disabled", "alone", "today")),
    (Urgency.medium, ("this week", "soon", "blocked", "debris")),
]
_SKILLS = {
    "first aid": ("injur", "bleed", "first aid", "fell", "hurt"),
    "heavy lifting": ("debris", "fallen tree", "branch", "move furniture", "sandbag", "lift"),
    "transportation": ("ride", "drive", "pharmacy", "appointment", "groceries"),
    "welfare check": ("check on", "welfare", "hasn't answered", "alone", "elderly"),
    "translation": ("interpret", "translat", "doesn't speak english", "only speaks"),
    "home repair": ("leak", "broken window", "tarp", "roof"),
}
_CERTS = {"first aid": ("injur", "bleed", "unconscious", "fell")}
_LANGUAGES = ("spanish", "vietnamese", "somali", "chinese", "cantonese", "mandarin", "tagalog", "amharic", "korean")
_CATEGORIES = {
    "welfare check": ("check on", "welfare", "hasn't answered"),
    "debris removal": ("debris", "fallen tree", "branch"),
    "transportation": ("ride", "pharmacy", "appointment"),
    "home damage": ("leak", "roof", "broken window", "flood"),
    "medical assistance": ("injur", "medication", "bleed"),
}


class HeuristicParser:
    def parse(self, request: ServiceRequest) -> ParsedRequest:
        text = request.text.lower()

        def hits(terms: tuple[str, ...]) -> bool:
            return any(t in text for t in terms)

        urgency = next((u for u, terms in _URGENCY if hits(terms)), Urgency.low)
        skills = [s for s, terms in _SKILLS.items() if hits(terms)]
        certs = [c for c, terms in _CERTS.items() if hits(terms)]
        langs = [lang for lang in _LANGUAGES if lang in text]
        if langs and "translation" not in skills:
            skills.append("translation")
        category = request.category_hint or next(
            (c for c, terms in _CATEGORIES.items() if hits(terms)), "general assistance"
        )
        words = re.findall(r"[a-z]{4,}", text)
        keywords = list(dict.fromkeys(w for w in words if w not in _STOPWORDS))[:12]
        return ParsedRequest(
            category=category,
            urgency=urgency,
            keywords=keywords,
            required_skills=skills,
            required_certifications=certs,
            languages=langs,
            location_text=None,
            summary=request.text.strip().split(".")[0][:200],
        )


_STOPWORDS = {
    "that", "this", "with", "have", "from", "they", "their", "there", "been", "will", "would",
    "could", "please", "need", "needs", "help", "someone", "about", "into", "after", "since",
    "what", "when", "where", "which", "while", "also", "just", "very", "only",
}
