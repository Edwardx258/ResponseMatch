"""Compare matching modes on a labelled set: does the dual track cut false positives
compared with the LLM-only baseline?

Each line of the labelled file is:
    {"request": {"id": ..., "text": ..., "lat": ..., "lon": ...},
     "acceptable": ["v03", "v07"]}        # volunteers a dispatcher would accept

A recommendation outside "acceptable" counts as a false positive.

    uv run python eval/run_eval.py --labels eval/labeled.jsonl

The heuristic backend only checks that the harness runs; its numbers say nothing
about the system, so they are printed with a warning and are not results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from responsematch.combine import combine
from responsematch.config import MODES, Settings
from responsematch.graph import Deps, build_deps, build_graph
from responsematch.schemas import ServiceRequest


@dataclass
class ModeStats:
    recommended: int = 0
    false_positives: int = 0
    true_positives: int = 0
    acceptable: int = 0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.recommended if self.recommended else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.acceptable if self.acceptable else 0.0


def load_labels(path: Path) -> list[tuple[ServiceRequest, set[str]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows.append((ServiceRequest.model_validate(item["request"]), set(item["acceptable"])))
    return rows


def evaluate(deps: Deps, labels: list[tuple[ServiceRequest, set[str]]], modes=MODES) -> dict[str, ModeStats]:
    graph = build_graph(deps)
    stats = {m: ModeStats() for m in modes}
    for request, acceptable in labels:
        # run the graph (and its LLM calls) once, then re-merge the two tracks per mode
        state = graph.invoke({"request": request})
        for mode in modes:
            matches = combine(state["candidates"], state["rule_scores"], state["judgements"], deps.settings, mode)
            picked = {m.volunteer_id for m in matches if m.recommended}
            s = stats[mode]
            s.recommended += len(picked)
            s.true_positives += len(picked & acceptable)
            s.false_positives += len(picked - acceptable)
            s.acceptable += len(acceptable)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("eval/labeled.jsonl"))
    args = ap.parse_args()

    settings = Settings()
    labels = load_labels(args.labels)

    deps = build_deps(settings)
    stats = evaluate(deps, labels)

    if settings.backend == "heuristic":
        print("WARNING: heuristic backend - these numbers only show the harness runs. Not results.\n")
    model = getattr(getattr(deps.parser, "llm", None), "model", "-")
    print(f"{len(labels)} labelled requests, backend={settings.backend}, model={model}\n")
    print(f"{'mode':<12}{'recommended':>12}{'false pos':>11}{'FP rate':>9}{'recall':>8}")
    for mode, s in stats.items():
        print(f"{mode:<12}{s.recommended:>12}{s.false_positives:>11}{s.false_positive_rate:>9.2f}{s.recall:>8.2f}")
    base, dual = stats.get("llm_only"), stats.get("dual_track")
    if base and dual and base.false_positive_rate:
        change = 1 - dual.false_positive_rate / base.false_positive_rate
        print(f"\nFalse-positive rate, dual_track vs llm_only: {change:+.0%} relative reduction")


if __name__ == "__main__":
    main()
