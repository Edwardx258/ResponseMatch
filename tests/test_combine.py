from responsematch.combine import combine
from responsematch.schemas import Judgement, RuleScore, Volunteer


def setup():
    vols = [
        Volunteer(id="near", name="Near", skills=[], lat=0, lon=0),
        Volunteer(id="far", name="Far", skills=[], lat=0, lon=0),
    ]
    rules = {
        "near": RuleScore(volunteer_id="near", score=0.8, eligible=True, distance_km=1, reasons=[]),
        "far": RuleScore(volunteer_id="far", score=0.1, eligible=False, distance_km=30, reasons=["too far"]),
    }
    llm = {
        "near": Judgement(volunteer_id="near", score=0.7, rationale="good"),
        "far": Judgement(volunteer_id="far", score=0.95, rationale="great skills"),
    }
    return vols, rules, llm


def picked(matches):
    return {m.volunteer_id for m in matches if m.recommended}


def test_llm_only_trusts_the_judge(settings):
    assert picked(combine(*setup(), settings, "llm_only")) == {"near", "far"}


def test_dual_track_drops_what_the_rules_exclude(settings):
    assert picked(combine(*setup(), settings, "dual_track")) == {"near"}


def test_weighted_respects_eligibility(settings):
    assert picked(combine(*setup(), settings, "weighted")) == {"near"}


def test_recommended_sort_first(settings):
    matches = combine(*setup(), settings, "dual_track")
    assert [m.volunteer_id for m in matches] == ["near", "far"]
