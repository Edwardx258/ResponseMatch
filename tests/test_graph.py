from responsematch.graph import build_graph, run_match


def test_injury_request_goes_to_certified_nearby_volunteer(deps, sample_requests):
    result = run_match(build_graph(deps), sample_requests["r04"], "dual_track")
    assert result.parsed.required_certifications == ["first aid"]
    picked = [m.volunteer_id for m in result.matches if m.recommended]
    assert picked == ["v04"]  # West Seattle, first aid certified
    assert result.passages and result.passages[0].source == "first-aid.md"


def test_language_need_is_a_hard_constraint(deps, sample_requests):
    result = run_match(build_graph(deps), sample_requests["r03"], "dual_track")
    picked = {m.volunteer_id for m in result.matches if m.recommended}
    assert picked == {"v03"}  # the only Vietnamese speaker in range


def test_llm_only_recommends_more_than_dual_track(deps, sample_requests):
    graph = build_graph(deps)
    for request in sample_requests.values():
        dual = {m.volunteer_id for m in run_match(graph, request, "dual_track").matches if m.recommended}
        llm = {m.volunteer_id for m in run_match(graph, request, "llm_only").matches if m.recommended}
        assert dual <= llm
