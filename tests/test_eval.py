import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_eval import evaluate  # noqa: E402


def test_evaluate_counts(deps, sample_requests):
    labels = [(sample_requests["r04"], {"v04"}), (sample_requests["r03"], {"v03"})]
    stats = evaluate(deps, labels, modes=("dual_track", "llm_only"))
    dual, llm = stats["dual_track"], stats["llm_only"]
    assert dual.true_positives == 2 and dual.false_positives == 0
    assert llm.recommended >= dual.recommended
    assert llm.false_positives == llm.recommended - llm.true_positives
