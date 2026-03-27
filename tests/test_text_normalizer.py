from __future__ import annotations

import unittest

from app.services.text_normalizer import (
    TextNormalizationConfig,
    TextSource,
    get_default_config_for_source,
    normalize_record,
    normalize_text,
    normalize_transcript_payload,
)


class TestTextNormalizationChanges(unittest.TestCase):
    def test_changes_include_expected_steps_for_transcription(self) -> None:
        cfg = get_default_config_for_source(TextSource.TRANSCRIPTION)
        result = normalize_text("https://x.test \U0001f624  UH   Hello!!!  ", cfg)
        self.assertNotEqual(result.original_text, result.normalized_text)
        actions = [c.action for c in result.changes]
        self.assertIn("strip_urls", actions)
        self.assertIn("strip_emoji", actions)
        self.assertIn("strip_whitespace", actions)
        self.assertIn("collapse_whitespace", actions)
        self.assertIn("collapse_repeated_punctuation", actions)
        self.assertIn("normalize_fillers", actions)
        self.assertIn("lowercase", actions)
        self.assertTrue(all(c.field == "text" for c in result.changes))

    def test_each_change_records_before_after(self) -> None:
        cfg = get_default_config_for_source(TextSource.TASK_ANSWER)
        result = normalize_text("  ABC  ", cfg)
        lower = [c for c in result.changes if c.action == "lowercase"]
        self.assertEqual(len(lower), 1)
        self.assertEqual(lower[0].before, "ABC")
        self.assertEqual(lower[0].after, "abc")

    def test_language_hint_latin(self) -> None:
        cfg = get_default_config_for_source(TextSource.TASK_ANSWER)
        result = normalize_text("Cannot find the button", cfg)
        self.assertEqual(result.language_hint, "en")

    def test_normalize_record_uses_transcription_defaults(self) -> None:
        rec = normalize_record("id-1", "  Um, OK  ", "think_aloud")
        self.assertEqual(rec.source, TextSource.TRANSCRIPTION.value)
        self.assertEqual(rec.text, ", ok")
        self.assertIn("language_hint", rec.metadata)
        self.assertIn("normalization_change_count", rec.metadata)
        self.assertEqual(rec.metadata["text_source"], TextSource.TRANSCRIPTION.value)

    def test_survey_newlines_and_numbering(self) -> None:
        cfg = get_default_config_for_source(TextSource.TASK_ANSWER)
        r = normalize_text("1. The page was slow\n2. I was lost", cfg)
        self.assertNotIn("\n", r.normalized_text)
        self.assertIn("slow", r.normalized_text)
        self.assertTrue(any(c.action == "strip_survey_enumeration" for c in r.changes))

    def test_study_abbreviation_expansion(self) -> None:
        cfg = get_default_config_for_source(TextSource.STUDY_TEXT)
        r = normalize_text("Poor UX and confusing UI.", cfg)
        self.assertIn("user experience", r.normalized_text)
        self.assertIn("user interface", r.normalized_text)

    def test_contraction_expansion_after_lowercase(self) -> None:
        cfg = get_default_config_for_source(TextSource.TASK_ANSWER)
        r = normalize_text("I Didn't understand the label.", cfg)
        self.assertIn("did not understand", r.normalized_text)
        self.assertTrue(any(c.action == "expand_contractions" for c in r.changes))

    def test_transcription_like_and_you_know(self) -> None:
        cfg = get_default_config_for_source(TextSource.TRANSCRIPTION)
        r = normalize_text("like, you know, it was slow", cfg)
        self.assertNotIn("like", r.normalized_text.split())
        self.assertNotIn("you", r.normalized_text)
        self.assertIn("slow", r.normalized_text)


class TestPayloadNormalization(unittest.TestCase):
    def test_repairs_typo_aliases_and_coerces_timestamps(self) -> None:
        payload = {
            "transcipt": "  Hello!!!  ",
            "start": "12.4",
            "end": 18,
            "regions": [],
            "regionsCount": 99,
        }
        out = normalize_transcript_payload(payload)
        p = out.normalized_payload
        self.assertEqual(p["transcript"], "hello!!")
        self.assertEqual(p["startTimeSec"], 12.4)
        self.assertEqual(p["endTimeSec"], 18.0)
        self.assertEqual(p["regions"], [])
        self.assertEqual(p["regionsCount"], 0)

        actions = [c.action for c in out.changes]
        self.assertIn("repair_field_alias", actions)
        self.assertIn("normalize_timestamp", actions)
        self.assertIn("recompute_regions_count", actions)
        self.assertIn("normalize_text", actions)

    def test_regions_normalized_and_count_recomputed(self) -> None:
        payload = {
            "text": "top",
            "start": "0",
            "end": "10",
            "regions": [
                {"utterance": "  Hmm, yes  ", "start_sec": "1.0", "end_sec": "2.5"},
            ],
            "regionsCount": 0,
        }
        # Task-answer profile: no ASR filler stripping so we assert region text + aliases only.
        text_cfg = TextNormalizationConfig(
            source=TextSource.TASK_ANSWER,
            normalize_fillers=False,
        )
        out = normalize_transcript_payload(payload, text_config=text_cfg)
        p = out.normalized_payload
        self.assertEqual(len(p["regions"]), 1)
        self.assertEqual(p["regions"][0]["transcript"], "hmm, yes")
        self.assertEqual(p["regions"][0]["startTimeSec"], 1.0)
        self.assertEqual(p["regions"][0]["endTimeSec"], 2.5)
        self.assertEqual(p["regionsCount"], 1)

        self.assertTrue(
            any(c.action == "normalize_regions" for c in out.changes),
            "expected aggregate regions change when list content changes",
        )

    def test_missing_transcript_field_emits_warning(self) -> None:
        payload = {
            "start": 0.0,
            "end": 1.0,
            "regions": [],
        }
        out = normalize_transcript_payload(payload)
        self.assertTrue(
            any("Missing transcript" in w for w in out.warnings),
        )
        self.assertEqual(out.normalized_payload["transcript"], "")

    def test_original_payload_preserved_on_result(self) -> None:
        payload = {"transcipt": "x", "start": "1", "end": "2", "regions": []}
        out = normalize_transcript_payload(payload)
        self.assertEqual(out.original_payload["transcipt"], "x")
        self.assertIn("transcript", out.normalized_payload)


if __name__ == "__main__":
    unittest.main()
