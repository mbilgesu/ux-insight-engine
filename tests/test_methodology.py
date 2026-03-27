from __future__ import annotations

import unittest

from app.services.explainability import explain
from app.services.nasa_tlx_analyzer import nasa_tlx_unweighted, nasa_tlx_weighted
from app.services.nielsen_mapper import map_issue_to_nielsen
from app.services.semantic_analysis import analyze_record
from app.services.sus_analyzer import sus_adjective_rating, sus_score
from app.services.text_normalizer import NormalizedRecord


class TestSUS(unittest.TestCase):
    def test_all_neutral_midpoint(self) -> None:
        # All 3 -> contributes 2 per item * 10 = 20, * 2.5 = 50
        score = sus_score([3] * 10)
        self.assertEqual(score, 50.0)
        self.assertEqual(sus_adjective_rating(score), "poor")

    def test_perfect_positive(self) -> None:
        # Odd: 5 -> 4 each * 5 = 20; Even: 5 -> 0 each... wait even items 5 -> (5-5)=0, odd 5 -> 4
        # Items 1,3,5,7,9: score 5 -> 4 each = 20; items 2,4,6,8,10: score 5 -> 0 = 0; total 20 * 2.5 = 50
        # For max: odd items 5 -> 4, even items 1 -> 4, so each pair 8, 5 pairs = 40, * 2.5 = 100
        r = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
        self.assertEqual(sus_score(r), 100.0)


class TestNASATLX(unittest.TestCase):
    def test_unweighted(self) -> None:
        self.assertEqual(nasa_tlx_unweighted([0, 0, 0, 100, 0, 0]), 16.67)

    def test_weighted(self) -> None:
        # All 50, weights equal -> 50
        dims = [50.0] * 6
        w = [100 / 6] * 6
        self.assertEqual(nasa_tlx_weighted(dims, w), 50.0)


class TestExplainability(unittest.TestCase):
    def test_explain_links_nielsen(self) -> None:
        rec = NormalizedRecord("1", "I could not find settings.", "survey")
        a = analyze_record(rec)
        n = map_issue_to_nielsen(a.issue_category)
        e = explain(a, n)
        self.assertIn("navigation", e.reasoning.lower())
        self.assertTrue(e.recommendation)


if __name__ == "__main__":
    unittest.main()
