from __future__ import annotations

import unittest

from semantic_analysis import analyze_text, normalize_and_analyze


class TestNormalizeAndSemanticPipeline(unittest.TestCase):
    def test_normalize_and_analyze_returns_record_and_analysis(self) -> None:
        rec, analysis = normalize_and_analyze(
            "r1",
            "I could not find the settings button.",
            "think_aloud",
        )
        self.assertEqual(rec.record_id, "r1")
        self.assertEqual(analysis.issue_category, "navigation")
        self.assertIsNotNone(rec.metadata.get("language_hint"))

    def test_analyze_text_matches_expanded_contraction(self) -> None:
        out = analyze_text("x", "I didn't understand the wording.", "survey")
        self.assertEqual(out.issue_category, "terminology")
