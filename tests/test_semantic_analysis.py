from __future__ import annotations

import unittest

from nielsen_mapper import map_issue_to_nielsen
from semantic_analysis import analyze_record, analyze_records
from text_normalizer import NormalizedRecord, normalize_record


class TestSemanticAnalysis(unittest.TestCase):
    def test_navigation_and_delay(self) -> None:
        r = NormalizedRecord(
            "1",
            "I had no idea where settings were. Took me a while to find.",
            "think_aloud",
        )
        out = analyze_record(r)
        self.assertEqual(out.issue_category, "navigation")
        self.assertEqual(out.task_signal, "task_delay")
        self.assertIn("navigation:took me a while to find", out.matched_rules)

    def test_negative_sentiment_override_positive(self) -> None:
        r = NormalizedRecord(
            "2",
            "It was easy to use but very confusing at the end.",
            "survey",
        )
        out = analyze_record(r)
        self.assertEqual(out.sentiment_label, "negative")

    def test_performance(self) -> None:
        r = NormalizedRecord("3", "The page was slow and laggy.", "survey")
        out = analyze_record(r)
        self.assertEqual(out.issue_category, "performance")

    def test_analyze_records_batch(self) -> None:
        rows = [
            NormalizedRecord("a", "nothing happened when I clicked", "survey"),
            NormalizedRecord("b", "clear and worked well", "survey"),
        ]
        outs = analyze_records(rows)
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0].issue_category, "feedback_missing")
        self.assertEqual(outs[1].sentiment_label, "positive")

    def test_normalize_then_analyze(self) -> None:
        raw = "  Um, I could NOT find   the button  "
        rec = normalize_record("x", raw, "think_aloud")
        self.assertNotIn("  ", rec.text)
        out = analyze_record(rec)
        self.assertEqual(out.issue_category, "navigation")


class TestNielsenMapper(unittest.TestCase):
    def test_maps_navigation(self) -> None:
        m = map_issue_to_nielsen("navigation")
        assert m is not None
        self.assertEqual(m.heuristics[0].number, 6)

    def test_none_category(self) -> None:
        self.assertIsNone(map_issue_to_nielsen(None))


if __name__ == "__main__":
    unittest.main()
