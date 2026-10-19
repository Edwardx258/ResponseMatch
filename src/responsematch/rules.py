"""Stage 3a: the rule track. Transparent, deterministic scoring with hard constraints."""

from __future__ import annotations

import math

from .schemas import URGENCY_LEVEL, GeoPoint, ParsedRequest, RuleScore, Volunteer

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _norm(items: list[str]) -> set[str]:
    return {i.strip().lower() for i in items if i.strip()}


def score_volunteer(parsed: ParsedRequest, location: GeoPoint | None, v: Volunteer) -> RuleScore:
    reasons: list[str] = []
    eligible = True

    # hard constraints
    if not v.available:
        eligible = False
        reasons.append("not available")
    missing_certs = _norm(parsed.required_certifications) - _norm(v.certifications)
    if missing_certs:
        eligible = False
        reasons.append(f"missing certification: {', '.join(sorted(missing_certs))}")
    missing_langs = _norm(parsed.languages) - _norm(v.languages)
    if missing_langs:
        eligible = False
        reasons.append(f"does not speak: {', '.join(sorted(missing_langs))}")

    distance = None
    proximity = 0.5  # neutral when the request has no location
    if location is not None:
        distance = haversine_km(location, GeoPoint(lat=v.lat, lon=v.lon))
        if distance > v.max_distance_km:
            eligible = False
            reasons.append(f"{distance:.1f} km away, beyond their {v.max_distance_km:.0f} km range")
        proximity = max(0.0, 1.0 - distance / max(v.max_distance_km, 0.1))

    # soft score: skill coverage and proximity; urgency shifts weight towards proximity
    need = _norm(parsed.required_skills)
    have = _norm(v.skills)
    coverage = len(need & have) / len(need) if need else 0.5
    if need & have:
        reasons.append(f"skills: {', '.join(sorted(need & have))}")
    w_prox = 0.3 + 0.1 * URGENCY_LEVEL[parsed.urgency]
    score = (1 - w_prox) * coverage + w_prox * proximity

    return RuleScore(
        volunteer_id=v.id,
        score=round(score, 4),
        eligible=eligible,
        distance_km=None if distance is None else round(distance, 2),
        reasons=reasons,
    )


def score_all(parsed: ParsedRequest, location: GeoPoint | None, volunteers: list[Volunteer]) -> dict[str, RuleScore]:
    return {v.id: score_volunteer(parsed, location, v) for v in volunteers}
