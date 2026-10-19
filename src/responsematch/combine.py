"""Stage 4: merge the two tracks into ranked recommendations.

Modes (compare them in eval/run_eval.py):
- dual_track: recommend only when the rules allow it AND both tracks clear their thresholds
- weighted:   recommend when the rules allow it and the weighted score clears one threshold
- llm_only:   the LLM judge alone (the baseline the dual track is meant to beat)
- rule_only:  the rule track alone
"""

from __future__ import annotations

from .config import Mode, Settings
from .schemas import Judgement, Match, RuleScore, Volunteer


def combine(
    volunteers: list[Volunteer],
    rule_scores: dict[str, RuleScore],
    judgements: dict[str, Judgement],
    settings: Settings,
    mode: Mode,
) -> list[Match]:
    matches = []
    for v in volunteers:
        r = rule_scores.get(v.id)
        j = judgements.get(v.id)
        rs = r.score if r else None
        ls = j.score if j else None
        eligible = bool(r and r.eligible)

        if mode == "llm_only":
            combined = ls or 0.0
            recommended = ls is not None and ls >= settings.llm_min
        elif mode == "rule_only":
            combined = rs or 0.0
            recommended = eligible and (rs or 0.0) >= settings.rule_min
        else:
            w = settings.rule_weight
            combined = w * (rs or 0.0) + (1 - w) * (ls or 0.0)
            if mode == "weighted":
                recommended = eligible and combined >= settings.weighted_min
            else:  # dual_track
                recommended = (
                    eligible
                    and rs is not None and rs >= settings.rule_min
                    and ls is not None and ls >= settings.llm_min
                )

        matches.append(
            Match(
                volunteer_id=v.id,
                name=v.name,
                recommended=recommended,
                combined_score=round(combined, 4),
                rule_score=rs,
                llm_score=ls,
                distance_km=r.distance_km if r else None,
                reasons=r.reasons if r else [],
                rationale=j.rationale if j else None,
                cited_passages=j.cited_passages if j else [],
            )
        )
    matches.sort(key=lambda m: (m.recommended, m.combined_score), reverse=True)
    return matches
