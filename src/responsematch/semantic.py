"""Stage 3b: the semantic track. An LLM judges each candidate against the request,
grounded in the retrieved handbook passages."""

from __future__ import annotations

import json
from typing import Protocol

from .claude import StructuredClaude
from .schemas import Judgement, Judgements, ParsedRequest, Passage, ServiceRequest, Volunteer

JUDGE_SYSTEM = """You help a volunteer dispatcher decide which volunteers fit a community \
service request. For every candidate, give a fit score from 0 to 1 and one sentence of \
rationale.

Base the judgement on the request, the candidate's profile, and the handbook passages \
provided. Cite the ids of passages you relied on. If the handbooks say a task needs \
training or should go to professionals, score untrained candidates low. Judge only the \
candidates listed and return one judgement per candidate."""


class Judge(Protocol):
    def judge(
        self, request: ServiceRequest, parsed: ParsedRequest, candidates: list[Volunteer], passages: list[Passage]
    ) -> dict[str, Judgement]: ...


def _clamp(j: Judgement) -> Judgement:
    return j.model_copy(update={"score": min(1.0, max(0.0, j.score))})


class ClaudeJudge:
    def __init__(self, claude: StructuredClaude):
        self.claude = claude

    def judge(self, request, parsed, candidates, passages):
        if not candidates:
            return {}
        profiles = [
            v.model_dump(include={"id", "skills", "languages", "certifications", "available"}) for v in candidates
        ]
        handbook = "\n\n".join(f'<passage id="{p.id}" source="{p.source}">\n{p.text}\n</passage>' for p in passages)
        prompt = (
            f"<request>\n{request.text}\n</request>\n"
            f"<parsed>\n{parsed.model_dump_json(indent=2)}\n</parsed>\n"
            f"<handbook>\n{handbook or '(no passages found)'}\n</handbook>\n"
            f"<candidates>\n{json.dumps(profiles, indent=2)}\n</candidates>"
        )
        result = self.claude.call(system=JUDGE_SYSTEM, prompt=prompt, schema=Judgements, effort="medium")
        wanted = {v.id for v in candidates}
        return {j.volunteer_id: _clamp(j) for j in result.judgements if j.volunteer_id in wanted}


class KeywordJudge:
    """Offline baseline: overlap between the request's needs and each profile.
    Deliberately ignores distance and hard constraints, like a naive LLM-only matcher."""

    def judge(self, request, parsed, candidates, passages):
        need = {s.lower() for s in parsed.required_skills + parsed.keywords}
        cited = [p.id for p in passages[:1]]
        out = {}
        for v in candidates:
            profile = {s.lower() for s in v.skills + v.languages + v.certifications}
            words = {w for item in profile for w in item.split()}
            overlap = len(need & (profile | words))
            score = min(1.0, 0.3 + 0.25 * overlap) if need else 0.5
            out[v.id] = Judgement(
                volunteer_id=v.id,
                score=round(score, 4),
                rationale=f"{overlap} of the request's needs appear in the profile",
                cited_passages=cited,
            )
        return out
