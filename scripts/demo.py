"""Run every sample request through the workflow and print who gets recommended.

    RESPONSEMATCH_BACKEND=heuristic uv run python scripts/demo.py   # offline
    uv run python scripts/demo.py                                    # Claude (needs a key)
"""

from __future__ import annotations

import json

from responsematch.config import ROOT, Settings
from responsematch.graph import build_deps, build_graph, run_match
from responsematch.schemas import ServiceRequest


def main() -> None:
    settings = Settings()
    deps = build_deps(settings)
    graph = build_graph(deps)
    lines = (ROOT / "data/sample/requests.jsonl").read_text(encoding="utf-8").splitlines()
    for line in filter(None, map(str.strip, lines)):
        request = ServiceRequest.model_validate(json.loads(line))
        result = run_match(graph, request, settings.mode)
        p = result.parsed
        print(f"\n[{request.id}] {request.text}")
        print(f"  parsed: {p.category} | urgency={p.urgency.value} | skills={p.required_skills} "
              f"| certs={p.required_certifications} | languages={p.languages}")
        print(f"  handbook: {', '.join(x.id for x in result.passages)}")
        for m in result.matches[:3]:
            mark = "✓" if m.recommended else "·"
            print(f"  {mark} {m.volunteer_id} combined={m.combined_score:.2f} rule={m.rule_score} "
                  f"llm={m.llm_score} {'; '.join(m.reasons)}")
    print(f"\nbackend={settings.backend} mode={settings.mode}")


if __name__ == "__main__":
    main()
