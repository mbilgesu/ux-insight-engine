"""
text_normalizer.py
==================
NLP preprocessing + payload schema repair service 

Two responsibilities, intentionally kept in one module:

  1. TEXT NORMALIZATION
     Cleans raw strings coming from three RUXAILAB data sources:
       - TaskAnswer.taskAnswer / taskObservations      → TextSource.TASK_ANSWER
       - Transcription segments (ASR output)           → TextSource.TRANSCRIPTION
       - Study.testDescription / studyConclusion, etc  → TextSource.STUDY_TEXT

     Each source gets source-aware defaults (e.g. filler removal is ON for
     transcriptions, OFF for researcher-authored text).

  2. PAYLOAD NORMALIZATION
     Repairs inconsistent Firestore payloads before they reach the NLP layer:
       - Field alias resolution  ("transcipt" → "transcript", "start" → "startTimeSec")
       - Timestamp coercion      ("12.4" → 12.4, None → None)
       - Region list integrity   (ensures regions list + recomputes regionsCount)

     Every mutation is recorded as a NormalizationChange so the pipeline
     remains fully auditable — satisfying the project's Explainable AI requirement
     at the data layer, not just the inference layer.

LANGUAGE SUPPORT
  RUXAILAB supports 10 languages (en, pt_br, es, fr, de, ar, hi, ja, ru, zh).
  Language detection is heuristic-only here (Unicode script ranges, no heavy deps).
  The language_hint on TextNormalizationResult is a routing signal for downstream
  modules; accurate detection should use langdetect / lingua-py at that layer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# Public surface
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "TextSource",
    "TextNormalizationConfig",
    "PayloadNormalizationConfig",
    "NormalizationChange",
    "TextNormalizationResult",
    "PayloadNormalizationResult",
    "normalize_text",
    "normalize_transcript_payload",
    "get_default_config_for_source",
    "NormalizedRecord",
    "normalize_record",
]


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class TextSource(str, Enum):
    """
    Maps to the three text origins in RUXAILAB's data model.

    Downstream modules can branch on this value to apply source-specific
    behaviour without needing to inspect the text itself.
    """

    TASK_ANSWER = "task_answer"
    TRANSCRIPTION = "transcription"
    STUDY_TEXT = "study_text"


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TextNormalizationConfig:
    """
    Controls every text-cleaning step.  Frozen so configs can be cached and
    shared safely across threads.

    source
        Which RUXAILAB data surface this config is for.  Used by
        get_default_config_for_source() to choose sensible defaults.
        Does not affect logic inside normalize_text() directly — that lets
        callers override individual flags while keeping source metadata.

    normalize_fillers
        Strip ASR filler words ("um", "uh", etc.).  Default OFF; the
        source-aware factory sets it ON for TRANSCRIPTION.
    """

    source: TextSource = TextSource.TASK_ANSWER

    normalize_unicode: bool = True
    remove_zero_width_chars: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True

    strip_whitespace: bool = True
    collapse_internal_whitespace: bool = True

    collapse_repeated_punctuation: bool = True
    max_repeated_punctuation: int = 2
    trim_punctuation_edges: bool = False

    lowercase: bool = True
    normalize_fillers: bool = False
    filler_words: Tuple[str, ...] = ("um", "uh", "erm", "hmm", "er", "ah")
    filler_phrases: Tuple[str, ...] = ()
    expand_contractions: bool = True
    collapse_newlines_to_space: bool = False
    strip_survey_enumeration: bool = False
    expand_study_abbreviations: bool = False


@dataclass(frozen=True)
class PayloadNormalizationConfig:
    """
    Defines canonical field names and the aliases that RUXAILAB's Firestore
    documents use in practice (including known typos such as "transcipt").
    """

    repair_common_field_typos: bool = True

    canonical_text_field: str = "transcript"
    allowed_text_aliases: Tuple[str, ...] = (
        "transcript",
        "text",
        "utterance",
        "content",
        "transcipt",
    )

    canonical_start_field: str = "startTimeSec"
    allowed_start_aliases: Tuple[str, ...] = (
        "startTimeSec",
        "start",
        "start_sec",
        "start_time",
    )

    canonical_end_field: str = "endTimeSec"
    allowed_end_aliases: Tuple[str, ...] = (
        "endTimeSec",
        "end",
        "end_sec",
        "end_time",
    )

    canonical_regions_field: str = "regions"
    allowed_regions_aliases: Tuple[str, ...] = (
        "regions",
        "segments",
        "chunks",
    )

    ensure_regions_list: bool = True
    compute_regions_count: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Result / audit types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NormalizationChange:
    field: str
    action: str
    before: Any
    after: Any


@dataclass
class TextNormalizationResult:
    original_text: str
    normalized_text: str
    language_hint: Optional[str] = None
    source: TextSource = TextSource.TASK_ANSWER
    changes: List[NormalizationChange] = field(default_factory=list)


@dataclass
class PayloadNormalizationResult:
    original_payload: Dict[str, Any]
    normalized_payload: Dict[str, Any]
    changes: List[NormalizationChange] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex constants
# ─────────────────────────────────────────────────────────────────────────────

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_MULTISPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")

_SURVEY_ENUM_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+|[a-z])[.)]\s+",
    re.IGNORECASE | re.MULTILINE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
_ADJACENT_DUP_RE = re.compile(r"\b(\w+)( \1\b)+", re.IGNORECASE)

# Longer / specific contractions first (avoid partial replacements)
_CONTRACTION_SPECS: Tuple[Tuple[str, str], ...] = (
    (r"\bwon't\b", "will not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"),
    (r"\bcan't\b", "can not"),
    (r"\bisn't\b", "is not"),
    (r"\bwasn't\b", "was not"),
    (r"\baren't\b", "are not"),
    (r"\bweren't\b", "were not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhadn't\b", "had not"),
    (r"\bit's\b", "it is"),
    (r"\bthat's\b", "that is"),
    (r"\bwhat's\b", "what is"),
    (r"\bwho's\b", "who is"),
    (r"\bwhere's\b", "where is"),
    (r"\bhere's\b", "here is"),
    (r"\bthere's\b", "there is"),
    (r"\blet's\b", "let us"),
    (r"\bi'm\b", "i am"),
    (r"\bwe're\b", "we are"),
    (r"\byou're\b", "you are"),
    (r"\bthey're\b", "they are"),
    (r"\bi've\b", "i have"),
    (r"\bwe've\b", "we have"),
    (r"\byou've\b", "you have"),
    (r"\bthey've\b", "they have"),
    (r"\bi'll\b", "i will"),
    (r"\bwe'll\b", "we will"),
    (r"\byou'll\b", "you will"),
    (r"\bthey'll\b", "they will"),
)

_CONTRACTION_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), r) for p, r in _CONTRACTION_SPECS
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE = re.compile(r"[a-zA-ZÀ-ÿ]")


# ─────────────────────────────────────────────────────────────────────────────
# Source-aware config factory
# ─────────────────────────────────────────────────────────────────────────────


def get_default_config_for_source(source: TextSource) -> TextNormalizationConfig:
    if source == TextSource.TRANSCRIPTION:
        return TextNormalizationConfig(
            source=TextSource.TRANSCRIPTION,
            normalize_fillers=True,
            filler_words=("um", "uh", "erm", "hmm", "er", "ah", "like"),
            filler_phrases=("you know", "i mean"),
        )
    if source == TextSource.TASK_ANSWER:
        return TextNormalizationConfig(
            source=TextSource.TASK_ANSWER,
            collapse_newlines_to_space=True,
            strip_survey_enumeration=True,
        )
    if source == TextSource.STUDY_TEXT:
        return TextNormalizationConfig(
            source=TextSource.STUDY_TEXT,
            expand_study_abbreviations=True,
        )
    return TextNormalizationConfig(source=source)


# ─────────────────────────────────────────────────────────────────────────────
# Public: normalize_text()
# ─────────────────────────────────────────────────────────────────────────────


def normalize_text(
    text: Optional[str],
    config: Optional[TextNormalizationConfig] = None,
) -> TextNormalizationResult:
    config = config or TextNormalizationConfig()
    current = _ensure_string(text)
    original = current
    changes: List[NormalizationChange] = []

    current = _apply(current, lambda t: _URL_RE.sub(" ", t), "text", "strip_urls", changes)
    current = _apply(current, lambda t: _EMOJI_RE.sub(" ", t), "text", "strip_emoji", changes)

    if config.remove_zero_width_chars:
        current = _apply(
            current,
            _remove_zero_width_chars,
            "text",
            "remove_zero_width_chars",
            changes,
        )

    if config.normalize_unicode:
        current = _apply(current, _normalize_unicode, "text", "normalize_unicode", changes)

    if config.normalize_quotes or config.normalize_dashes:

        def _qd(t: str) -> str:
            return _normalize_quotes_and_dashes(
                t,
                normalize_quotes=config.normalize_quotes,
                normalize_dashes=config.normalize_dashes,
            )

        current = _apply(current, _qd, "text", "normalize_quotes_dashes", changes)

    if config.collapse_newlines_to_space:
        current = _apply(
            current,
            _collapse_newlines_to_space,
            "text",
            "collapse_newlines_to_space",
            changes,
        )

    if config.strip_survey_enumeration:
        current = _apply(
            current,
            _strip_survey_enumeration,
            "text",
            "strip_survey_enumeration",
            changes,
        )

    if config.strip_whitespace:
        current = _apply(current, str.strip, "text", "strip_whitespace", changes)

    if config.collapse_internal_whitespace:
        current = _apply(current, _collapse_whitespace, "text", "collapse_whitespace", changes)

    if config.collapse_repeated_punctuation:

        def _rp(t: str) -> str:
            return _collapse_repeated_punctuation(t, config.max_repeated_punctuation)

        current = _apply(current, _rp, "text", "collapse_repeated_punctuation", changes)

    if config.expand_study_abbreviations:
        current = _apply(
            current,
            _expand_study_abbreviations,
            "text",
            "expand_study_abbreviations",
            changes,
        )

    if config.normalize_fillers:

        def _fil(t: str) -> str:
            return _normalize_fillers(t, config.filler_words)

        current = _apply(current, _fil, "text", "normalize_fillers", changes)

    if config.source == TextSource.TRANSCRIPTION:
        current = _apply(
            current,
            lambda t: _ADJACENT_DUP_RE.sub(r"\1", t),
            "text",
            "dedup_adjacent_tokens",
            changes,
        )

    if config.filler_phrases:

        def _fph(t: str) -> str:
            return _normalize_filler_phrases(t, config.filler_phrases)

        current = _apply(current, _fph, "text", "normalize_filler_phrases", changes)

    if config.trim_punctuation_edges:
        current = _apply(
            current,
            lambda t: t.strip(" \t\n\r.,!?;:"),
            "text",
            "trim_punctuation_edges",
            changes,
        )

    if config.lowercase:
        current = _apply(current, str.lower, "text", "lowercase", changes)

    if config.expand_contractions:

        def _ec(t: str) -> str:
            return _expand_contractions(t)

        current = _apply(current, _ec, "text", "expand_contractions", changes)

    if config.collapse_internal_whitespace:
        current = _apply(current, _collapse_whitespace, "text", "collapse_whitespace_final", changes)

    language_hint = _detect_language(current)

    return TextNormalizationResult(
        original_text=original,
        normalized_text=current,
        language_hint=language_hint,
        source=config.source,
        changes=changes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: normalize_transcript_payload()
# ─────────────────────────────────────────────────────────────────────────────


def normalize_transcript_payload(
    payload: Dict[str, Any],
    text_config: Optional[TextNormalizationConfig] = None,
    payload_config: Optional[PayloadNormalizationConfig] = None,
) -> PayloadNormalizationResult:
    text_config = text_config or get_default_config_for_source(TextSource.TRANSCRIPTION)
    payload_config = payload_config or PayloadNormalizationConfig()

    original_payload = _safe_dict_copy(payload)
    changes: List[NormalizationChange] = []
    warnings: List[str] = []

    canonical, key_changes, key_warnings = _canonicalize_payload_keys(
        payload, payload_config
    )
    changes.extend(key_changes)
    warnings.extend(key_warnings)

    text_result = normalize_text(canonical.get("transcript"), text_config)
    canonical["transcript"] = text_result.normalized_text
    if text_result.original_text != text_result.normalized_text:
        changes.append(
            NormalizationChange(
                field="transcript",
                action="normalize_text",
                before=text_result.original_text,
                after=text_result.normalized_text,
            )
        )
    changes.extend(text_result.changes)

    for ts_field in ("startTimeSec", "endTimeSec"):
        raw_val = canonical.get(ts_field)
        coerced_val = _normalize_timestamp(raw_val)
        if raw_val != coerced_val:
            changes.append(
                NormalizationChange(
                    field=ts_field,
                    action="normalize_timestamp",
                    before=raw_val,
                    after=coerced_val,
                )
            )
        canonical[ts_field] = coerced_val

    raw_regions = canonical.get("regions")
    if raw_regions is None:
        canonical["regions"] = []
        changes.append(
            NormalizationChange(
                field="regions",
                action="coerce_none_to_empty_list",
                before=None,
                after=[],
            )
        )
    else:
        normalized_regions, region_warnings = _normalize_regions(
            raw_regions, text_config, payload_config
        )
        if raw_regions != normalized_regions:
            changes.append(
                NormalizationChange(
                    field="regions",
                    action="normalize_regions",
                    before=raw_regions,
                    after=normalized_regions,
                )
            )
        canonical["regions"] = normalized_regions
        warnings.extend(region_warnings)

    canonical, count_change = _ensure_regions_count(canonical)
    if count_change:
        changes.append(count_change)

    return PayloadNormalizationResult(
        original_payload=original_payload,
        normalized_payload=canonical,
        changes=changes,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private: language detection
# ─────────────────────────────────────────────────────────────────────────────


def _detect_language(text: str) -> Optional[str]:
    if not text:
        return None
    if _ARABIC_RE.search(text):
        return "ar"
    if _CJK_RE.search(text):
        return "zh"
    if _DEVANAGARI_RE.search(text):
        return "hi"
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _LATIN_RE.search(text):
        return "en"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Private: payload helpers
# ─────────────────────────────────────────────────────────────────────────────


def _canonicalize_payload_keys(
    payload: Dict[str, Any],
    cfg: PayloadNormalizationConfig,
) -> Tuple[Dict[str, Any], List[NormalizationChange], List[str]]:
    normalized: Dict[str, Any] = {}
    changes: List[NormalizationChange] = []
    warnings: List[str] = []

    field_map = [
        (cfg.canonical_text_field, cfg.allowed_text_aliases, ""),
        (cfg.canonical_start_field, cfg.allowed_start_aliases, None),
        (cfg.canonical_end_field, cfg.allowed_end_aliases, None),
        (cfg.canonical_regions_field, cfg.allowed_regions_aliases, []),
    ]

    for canonical_name, aliases, default in field_map:
        found_key = _pick_first_existing_key(payload, aliases)
        if found_key is not None:
            normalized[canonical_name] = payload[found_key]
            if found_key != canonical_name:
                changes.append(
                    NormalizationChange(
                        field=canonical_name,
                        action="repair_field_alias",
                        before=found_key,
                        after=canonical_name,
                    )
                )
        else:
            normalized[canonical_name] = default
            if default == "":
                warnings.append("Missing transcript/text field; defaulted to empty string.")

    if "regionsCount" in payload:
        normalized["regionsCount"] = payload["regionsCount"]

    return normalized, changes, warnings


def _normalize_regions(
    regions: Optional[Sequence[Dict[str, Any]]],
    text_config: TextNormalizationConfig,
    payload_config: PayloadNormalizationConfig,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if regions is None:
        return [], []
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
        return [], ["Regions field was not a valid list-like sequence; defaulted to []."]

    normalized_regions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            warnings.append(f"Region {idx} was not an object and was skipped.")
            continue

        region_canonical, _, region_warnings = _canonicalize_payload_keys(
            region, payload_config
        )
        warnings.extend(region_warnings)

        text_result = normalize_text(region_canonical.get("transcript"), text_config)
        start_time = _normalize_timestamp(region_canonical.get("startTimeSec"))
        end_time = _normalize_timestamp(region_canonical.get("endTimeSec"))

        if region_canonical.get("transcript") is None:
            warnings.append(f"Region {idx} transcript was null; normalized to empty string.")
        if start_time is None:
            warnings.append(f"Region {idx} has missing or unparseable startTimeSec.")
        if end_time is None:
            warnings.append(f"Region {idx} has missing or unparseable endTimeSec.")

        normalized_regions.append(
            {
                "transcript": text_result.normalized_text,
                "startTimeSec": start_time,
                "endTimeSec": end_time,
            }
        )

    return normalized_regions, warnings


def _ensure_regions_count(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[NormalizationChange]]:
    updated = dict(payload)
    before = updated.get("regionsCount")
    after = len(updated.get("regions", []))
    updated["regionsCount"] = after
    if before != after:
        return updated, NormalizationChange(
            field="regionsCount",
            action="recompute_regions_count",
            before=before,
            after=after,
        )
    return updated, None


def _normalize_timestamp(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if stripped else None
        except ValueError:
            return None
    return None


def _pick_first_existing_key(
    payload: Dict[str, Any],
    aliases: Sequence[str],
) -> Optional[str]:
    for key in aliases:
        if key in payload:
            return key
    return None


def _safe_dict_copy(payload: Dict[str, Any]) -> Dict[str, Any]:
    return dict(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Private: text transformation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _apply(
    text: str,
    fn,
    field_name: str,
    action: str,
    changes: List[NormalizationChange],
) -> str:
    updated = fn(text)
    if updated != text:
        changes.append(
            NormalizationChange(
                field=field_name,
                action=action,
                before=text,
                after=updated,
            )
        )
    return updated


def _ensure_string(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("…", "...")
    return normalized


def _remove_zero_width_chars(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text)


def _normalize_quotes_and_dashes(
    text: str,
    *,
    normalize_quotes: bool = True,
    normalize_dashes: bool = True,
) -> str:
    if normalize_quotes:
        text = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
    if normalize_dashes:
        text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def _collapse_whitespace(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text)


def _collapse_repeated_punctuation(text: str, max_repeat: int) -> str:
    if max_repeat < 1:
        return text
    return _REPEATED_PUNCT_RE.sub(lambda m: m.group(1) * max_repeat, text)


def _normalize_fillers(text: str, filler_words: Sequence[str]) -> str:
    if not filler_words:
        return text
    pattern = r"\b(" + "|".join(re.escape(w) for w in filler_words) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def _collapse_newlines_to_space(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", re.sub(r"[\r\n]+", " ", text)).strip()


def _strip_survey_enumeration(text: str) -> str:
    t = _SURVEY_ENUM_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", t).strip()


def _expand_study_abbreviations(text: str) -> str:
    out = text
    out = re.sub(r"\bUI\b", "user interface", out)
    out = re.sub(r"\bUX\b", "user experience", out)
    out = re.sub(r"\bbtw\b", "by the way", out, flags=re.IGNORECASE)
    out = re.sub(r"\be\.g\.\b", "for example", out, flags=re.IGNORECASE)
    out = re.sub(r"\bi\.e\.\b", "that is", out, flags=re.IGNORECASE)
    out = re.sub(r"\betc\.\b", "etcetera", out, flags=re.IGNORECASE)
    return out


def _normalize_filler_phrases(text: str, phrases: Sequence[str]) -> str:
    if not phrases:
        return text
    out = text
    for ph in sorted(phrases, key=len, reverse=True):
        parts = ph.split()
        if not parts:
            continue
        inner = r"\s+".join(re.escape(p) for p in parts)
        pat = re.compile(r"(?<!\w)" + inner + r"(?!\w)", re.IGNORECASE)
        out = pat.sub(" ", out)
    return _MULTISPACE_RE.sub(" ", out).strip()


def _expand_contractions(text: str) -> str:
    out = text
    for pat, repl in _CONTRACTION_PATTERNS:
        out = pat.sub(repl, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline compatibility (CLI `analyze.py` + semantic_analysis NormalizedRecord)
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_CLI_TO_SOURCE: dict[str, TextSource] = {
    "think_aloud": TextSource.TRANSCRIPTION,
    "survey": TextSource.TASK_ANSWER,
    "moderator_notes": TextSource.STUDY_TEXT,
    "unknown": TextSource.TASK_ANSWER,
}


@dataclass
class NormalizedRecord:
    """Stable row for semantic_analysis: normalized text plus audit metadata."""

    record_id: str
    text: str
    source: str = TextSource.TASK_ANSWER.value
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_record(
    record_id: str,
    raw_text: str,
    source: Union[str, TextSource] = "unknown",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> NormalizedRecord:
    if isinstance(source, TextSource):
        ts = source
    else:
        ts = _LEGACY_CLI_TO_SOURCE.get(source, TextSource.TASK_ANSWER)
    cfg = get_default_config_for_source(ts)
    result = normalize_text(raw_text, cfg)
    meta: Dict[str, Any] = {
        "language_hint": result.language_hint,
        "normalization_change_count": len(result.changes),
        "text_source": ts.value,
    }
    if extra_metadata:
        meta.update(extra_metadata)
    return NormalizedRecord(
        record_id=record_id,
        text=result.normalized_text,
        source=ts.value,
        metadata=meta,
    )


if __name__ == "__main__":
    examples = [
        (
            TextSource.TASK_ANSWER,
            "  I can\u2019t find the Submit button!! \U0001f624",
        ),
        (
            TextSource.TRANSCRIPTION,
            "Uh um so I I clicked here and like\u2026 nothing happened you know",
        ),
        (
            TextSource.STUDY_TEXT,
            "Nielsen H3 (User Control) scored poorly. See H7.",
        ),
        (
            TextSource.TASK_ANSWER,
            "\u0644\u0627 \u0623\u0633\u062a\u0637\u064a\u0639 \u0627\u0644\u0639\u062b\u0648\u0631",
        ),
        (
            TextSource.TASK_ANSWER,
            "\u9001\u4fe1\u30dc\u30bf\u30f3\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093",
        ),
    ]

    print("=" * 64)
    print("TEXT NORMALIZATION")
    print("=" * 64)
    for source, raw in examples:
        cfg = get_default_config_for_source(source)
        result = normalize_text(raw, cfg)
        print(f"SOURCE  : {result.source.value}")
        print(f"RAW     : {result.original_text!r}")
        print(f"NORM    : {result.normalized_text!r}")
        print(f"LANG    : {result.language_hint}")
        print(f"CHANGES : {len(result.changes)}")
        print("-" * 64)

    sample_payload = {
        "transcipt": "  Uh... I expected it on the top right!!! ",
        "start": "12.4",
        "end": 18,
        "regions": [
            {"start": "0.0", "end": "3.1", "text": "  Hmm, okay... "},
            {
                "start_sec": 3.1,
                "end_sec": 6.5,
                "utterance": "I can\u2019t find the button???",
            },
            {"startTimeSec": None, "endTimeSec": 8.2, "transcipt": None},
        ],
        "regionsCount": 99,
    }

    print("\n" + "=" * 64)
    print("PAYLOAD NORMALIZATION")
    print("=" * 64)
    presult = normalize_transcript_payload(sample_payload)
    print("NORMALIZED PAYLOAD:")
    for k, v in presult.normalized_payload.items():
        print(f"  {k}: {v}")
    print(f"\nCHANGES ({len(presult.changes)}):")
    for c in presult.changes:
        print(f"  [{c.action}] {c.field}: {c.before!r} → {c.after!r}")
    print(f"\nWARNINGS ({len(presult.warnings)}):")
    for w in presult.warnings:
        print(f"  - {w}")
