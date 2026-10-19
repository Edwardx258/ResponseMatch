import pytest

from responsematch.rules import haversine_km, score_volunteer
from responsematch.schemas import GeoPoint, ParsedRequest, Urgency, Volunteer


def parsed(**kw) -> ParsedRequest:
    base = dict(category="test", urgency=Urgency.medium, keywords=[], required_skills=[], summary="t")
    return ParsedRequest(**{**base, **kw})


def volunteer(**kw) -> Volunteer:
    base = dict(id="v", name="V", skills=["first aid"], lat=47.6, lon=-122.33, max_distance_km=5)
    return Volunteer(**{**base, **kw})


def test_haversine_one_degree_of_latitude():
    assert haversine_km(GeoPoint(lat=47.0, lon=-122.0), GeoPoint(lat=48.0, lon=-122.0)) == pytest.approx(111.2, abs=0.2)


def test_close_volunteer_with_matching_skill_is_eligible():
    r = score_volunteer(parsed(required_skills=["first aid"]), GeoPoint(lat=47.601, lon=-122.33), volunteer())
    assert r.eligible
    assert r.score > 0.8
    assert "skills: first aid" in r.reasons


@pytest.mark.parametrize(
    "request_kw, volunteer_kw, reason",
    [
        ({}, {"available": False}, "not available"),
        ({"required_certifications": ["first aid"]}, {}, "missing certification"),
        ({"languages": ["somali"]}, {"languages": ["english"]}, "does not speak"),
    ],
)
def test_hard_constraints_make_volunteer_ineligible(request_kw, volunteer_kw, reason):
    r = score_volunteer(parsed(**request_kw), None, volunteer(**volunteer_kw))
    assert not r.eligible
    assert any(reason in x for x in r.reasons)


def test_out_of_range_is_ineligible():
    far = GeoPoint(lat=47.7, lon=-122.33)  # ~11 km north, range is 5 km
    r = score_volunteer(parsed(), far, volunteer())
    assert not r.eligible
    assert r.distance_km == pytest.approx(11.1, abs=0.2)


def test_urgency_weights_proximity_more():
    loc = GeoPoint(lat=47.62, lon=-122.33)  # ~2.2 km away
    v = volunteer(skills=["cooking"])
    low = score_volunteer(parsed(urgency=Urgency.low, required_skills=["first aid"]), loc, v)
    crit = score_volunteer(parsed(urgency=Urgency.critical, required_skills=["first aid"]), loc, v)
    assert crit.score > low.score  # no skill match, so proximity is all that scores
