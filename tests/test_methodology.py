from __future__ import annotations

import unittest
from typing import Sequence

from explainability import ExplainabilityEngine, SemanticAnalyzer
from ruxailab_methodology import (
    TLX_DIMENSION_KEYS,
    TLXWeights,
    analyze_nasa_tlx,
    calculate_sus_score,
    sus_adjective_for_score,
)
from text_normalizer import normalize_record


def _sus_score(responses: list[int]) -> float:
    return calculate_sus_score(responses).score


def _sus_adjective_rating(score: float) -> str:
    return sus_adjective_for_score(score).lower()


def _nasa_tlx_unweighted(dimensions: Sequence[float]) -> float:
    if len(dimensions) != 6:
        raise ValueError("NASA-TLX requires exactly 6 dimension scores")
    d = {k: float(dimensions[i]) for i, k in enumerate(TLX_DIMENSION_KEYS)}
    return round(analyze_nasa_tlx(d, TLXWeights.uniform()).rtlx_score, 2)


def _nasa_tlx_weighted(dimensions: Sequence[float], weights: Sequence[float]) -> float:
    if len(dimensions) != 6 or len(weights) != 6:
        raise ValueError("dimensions and weights must each have length 6")
    d = {k: float(dimensions[i]) for i, k in enumerate(TLX_DIMENSION_KEYS)}
    if len({round(w, 6) for w in weights}) == 1:
        return round(analyze_nasa_tlx(d, TLXWeights.uniform()).overall_score, 2)
    raise ValueError(
        "Non-uniform TLX weights require pairwise win counts; use "
        "ruxailab_methodology.TLXWeights.from_wins and analyze_nasa_tlx(..., weights=...).",
    )


class TestSUS(unittest.TestCase):
    def test_all_neutral_midpoint(self) -> None:
        # All 3 -> contributes 2 per item * 10 = 20, * 2.5 = 50
        score = _sus_score([3] * 10)
        self.assertEqual(score, 50.0)
        self.assertEqual(_sus_adjective_rating(score), "poor")

    def test_perfect_positive(self) -> None:
        r = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
        self.assertEqual(_sus_score(r), 100.0)


class TestNASATLX(unittest.TestCase):
    def test_unweighted(self) -> None:
        self.assertEqual(_nasa_tlx_unweighted([0, 0, 0, 100, 0, 0]), 16.67)

    def test_weighted(self) -> None:
        dims = [50.0] * 6
        w = [100 / 6] * 6
        self.assertEqual(_nasa_tlx_weighted(dims, w), 50.0)


class TestExplainability(unittest.TestCase):
    def test_explain_links_nielsen(self) -> None:
        rec = normalize_record("1", "I could not find settings.", "survey")
        xa = SemanticAnalyzer().analyze(rec.text)
        ex = ExplainabilityEngine().explain(xa)
        self.assertEqual(xa.issue_category.value, "navigation")
        self.assertIn("navigation", ex.reasoning_chain_text().lower())
        self.assertTrue(ex.recommendation)
        self.assertIsNotNone(ex.nielsen_ref)


if __name__ == "__main__":
    unittest.main()
