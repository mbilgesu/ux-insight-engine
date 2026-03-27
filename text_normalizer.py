"""
text_normalizer.py
==================
NLP preprocessing + payload schema repair service for RUXAILAB.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Public surface
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enums
    "TextSource",
    # Config
    "TextNormalizationConfig",
    "PayloadNormalizationConfig",
    # Result types
    "NormalizationChange",
    "TextNormalizationResult",
    "PayloadNormalizationResult",
    # Functions
    "normalize_text",
    "normalize_transcript_payload",
    "get_default_config_for_source",
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
    TASK_ANSWER   = "task_answer"    # TaskAnswer.taskAnswer / taskObservations
    TRANSCRIPTION = "transcription"  # Transcription segments (ASR output)
    STUDY_TEXT    = "study_text"     # Study description / conclusion / postAnswer


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

    # ── Unicode & encoding ───────────────────────────────────────────────────
    normalize_unicode: bool = True
    remove_zero_width_chars: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True

    # ── Whitespace ───────────────────────────────────────────────────────────
    strip_whitespace: bool = True
    collapse_internal_whitespace: bool = True

    # ── Punctuation ──────────────────────────────────────────────────────────
    collapse_repeated_punctuation: bool = True
    max_repeated_punctuation: int = 2
    trim_punctuation_edges: bool = False

    # ── Content ──────────────────────────────────────────────────────────────
    lowercase: bool = True
    normalize_fillers: bool = False          # ON by default for TRANSCRIPTION only
    filler_words: Tuple[str, ...] = ("um", "uh", "erm", "hmm", "er", "ah")


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
        "transcipt",              # real typo present in RUXAILAB codebase
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
    """
    Records a single mutation made during normalization.

    Collected in result.changes so the full transformation history is
    available for debugging, UI display, and audit logging.
    """
    field:  str
    action: str
    before: Any
    after:  Any


@dataclass
class TextNormalizationResult:
    """
    Output of normalize_text().

    original_text
        Untouched input — always preserved so callers can show raw text
        alongside the cleaned version.
    normalized_text
        Clean, ready-for-NLP string.
    language_hint
        BCP-47 tag inferred from Unicode script ranges ("en", "ar", "zh", …).
        None when detection is inconclusive.  Treat as a hint, not a guarantee.
    source
        Which RUXAILAB data surface produced this text.
    changes
        Ordered list of every mutation applied; empty if text was already clean.
    """
    original_text:   str
    normalized_text: str
    language_hint:   Optional[str]       = None
    source:          TextSource          = TextSource.TASK_ANSWER
    changes:         List[NormalizationChange] = field(default_factory=list)


@dataclass
class PayloadNormalizationResult:
    """Output of normalize_transcript_payload()."""
    original_payload:   Dict[str, Any]
    normalized_payload: Dict[str, Any]
    changes:  List[NormalizationChange] = field(default_factory=list)
    warnings: List[str]                 = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex constants
# ─────────────────────────────────────────────────────────────────────────────

_ZERO_WIDTH_RE       = re.compile(r"[\u200B-\u200D\uFEFF]")
_MULTISPACE_RE       = re.compile(r"\s+")
_REPEATED_PUNCT_RE   = re.compile(r"([!?.,])\1{2,}")

# Unicode script ranges — used by _detect_language()
_CJK_RE        = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_ARABIC_RE     = re.compile(r"[\u0600-\u06ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_CYRILLIC_RE   = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE      = re.compile(r"[a-zA-ZÀ-ÿ]")


# ─────────────────────────────────────────────────────────────────────────────
# Source-aware config factory  (Version 1 addition #3)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_config_for_source(source: TextSource) -> TextNormalizationConfig:
    """
    Return a TextNormalizationConfig with sensible defaults for the given source.

    TRANSCRIPTION  → filler removal ON  (ASR output is noisy)
    TASK_ANSWER    → defaults           (participant free text)
    STUDY_TEXT     → defaults           (researcher-authored, generally cleaner)

    Callers can still override individual flags after calling this factory:

        cfg = get_default_config_for_source(TextSource.TRANSCRIPTION)
        # cfg.normalize_fillers is already True
    """
    if source == TextSource.TRANSCRIPTION:
        return TextNormalizationConfig(
            source=TextSource.TRANSCRIPTION,
            normalize_fillers=True,
        )
    return TextNormalizationConfig(source=source)


# ─────────────────────────────────────────────────────────────────────────────
# Public: normalize_text()
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(
    text: Optional[str],
    config: Optional[TextNormalizationConfig] = None,
) -> TextNormalizationResult:
    """
    Apply the full normalization pipeline to a single string.

    Each step only fires if its flag is True in config.
    Every mutation that changes the text is recorded in result.changes.

    Language detection runs unconditionally after all cleaning steps
    (cleaner text → more reliable script detection).
    """
    config  = config or TextNormalizationConfig()
    current = _ensure_string(text)
    original = current
    changes: List[NormalizationChange] = []

    # ── Step 1: Remove zero-width characters ──────────────────────────────
    if config.remove_zero_width_chars:
        current = _apply(current, _remove_zero_width_chars, "text",
                         "remove_zero_width_chars", changes)

    # ── Step 2: Unicode normalization (NFKC) ──────────────────────────────
    if config.normalize_unicode:
        current = _apply(current, _normalize_unicode, "text",
                         "normalize_unicode", changes)

    # ── Step 3: Quote + dash normalization ────────────────────────────────
    if config.normalize_quotes or config.normalize_dashes:
        fn = lambda t: _normalize_quotes_and_dashes(
            t,
            normalize_quotes=config.normalize_quotes,
            normalize_dashes=config.normalize_dashes,
        )
        current = _apply(current, fn, "text", "normalize_quotes_dashes", changes)

    # ── Step 4: Strip leading/trailing whitespace ─────────────────────────
    if config.strip_whitespace:
        current = _apply(current, str.strip, "text", "strip_whitespace", changes)

    # ── Step 5: Collapse internal whitespace ─────────────────────────────
    if config.collapse_internal_whitespace:
        current = _apply(current, _collapse_whitespace, "text",
                         "collapse_whitespace", changes)

    # ── Step 6: Collapse repeated punctuation ─────────────────────────────
    if config.collapse_repeated_punctuation:
        fn = lambda t: _collapse_repeated_punctuation(t, config.max_repeated_punctuation)
        current = _apply(current, fn, "text",
                         "collapse_repeated_punctuation", changes)

    # ── Step 7: Filler word normalization (ASR-specific) ──────────────────
    if config.normalize_fillers:
        fn = lambda t: _normalize_fillers(t, config.filler_words)
        current = _apply(current, fn, "text", "normalize_fillers", changes)

    # ── Step 8: Trim punctuation from edges ───────────────────────────────
    if config.trim_punctuation_edges:
        current = _apply(
            current,
            lambda t: t.strip(" \t\n\r.,!?;:"),
            "text", "trim_punctuation_edges", changes,
        )

    # ── Step 9: Lowercase ─────────────────────────────────────────────────
    if config.lowercase:
        current = _apply(current, str.lower, "text", "lowercase", changes)

    # ── Post-pipeline: language detection (Version 1 addition #2) ─────────
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
    """
    Full pipeline for a raw Transcription Firestore document.

    Steps:
      1. Canonicalize field names (resolve aliases, repair typos)
      2. Normalize the top-level transcript string
      3. Coerce startTimeSec / endTimeSec to float
      4. Normalize regions list (text + timestamps per segment)
      5. Recompute regionsCount

    Returns a PayloadNormalizationResult with the clean payload, a full
    change log, and any warnings raised during processing.
    """
    text_config    = text_config    or TextNormalizationConfig(source=TextSource.TRANSCRIPTION)
    payload_config = payload_config or PayloadNormalizationConfig()

    original_payload = _safe_dict_copy(payload)
    changes:  List[NormalizationChange] = []
    warnings: List[str]                 = []

    # Step 1 — field alias repair
    canonical, key_changes, key_warnings = _canonicalize_payload_keys(
        payload, payload_config
    )
    changes.extend(key_changes)
    warnings.extend(key_warnings)

    # Step 2 — text normalization
    text_result = normalize_text(canonical.get("transcript"), text_config)
    canonical["transcript"] = text_result.normalized_text
    if text_result.original_text != text_result.normalized_text:
        changes.append(NormalizationChange(
            field="transcript",
            action="normalize_text",
            before=text_result.original_text,
            after=text_result.normalized_text,
        ))
    changes.extend(text_result.changes)

    # Step 3 — timestamp coercion
    for ts_field in ("startTimeSec", "endTimeSec"):
        raw_val       = canonical.get(ts_field)
        coerced_val   = _normalize_timestamp(raw_val)
        if raw_val != coerced_val:
            changes.append(NormalizationChange(
                field=ts_field,
                action="normalize_timestamp",
                before=raw_val,
                after=coerced_val,
            ))
        canonical[ts_field] = coerced_val

    # Step 4 — regions list
    raw_regions = canonical.get("regions")
    if raw_regions is None:
        canonical["regions"] = []
        changes.append(NormalizationChange(
            field="regions",
            action="coerce_none_to_empty_list",
            before=None,
            after=[],
        ))
    else:
        normalized_regions, region_warnings = _normalize_regions(
            raw_regions, text_config, payload_config
        )
        if raw_regions != normalized_regions:
            changes.append(NormalizationChange(
                field="regions",
                action="normalize_regions",
                before=raw_regions,
                after=normalized_regions,
            ))
        canonical["regions"] = normalized_regions
        warnings.extend(region_warnings)

    # Step 5 — recompute regionsCount
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
# Private: language detection  (Version 1 addition #2)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_language(text: str) -> Optional[str]:
    """
    Heuristic language detection via Unicode script ranges.
    Returns a BCP-47 tag or None when inconclusive.

    Check order (most-distinctive scripts first to minimise false positives):
      Arabic → "ar"
      CJK    → "zh"  (covers ja / ko; callers can refine with a heavier lib)
      Devanagari → "hi"
      Cyrillic   → "ru"
      Latin      → "en"  (conservative default for all Latin-script languages)
    """
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
    """Resolve field aliases to canonical names and record each rename."""
    normalized: Dict[str, Any] = {}
    changes:    List[NormalizationChange] = []
    warnings:   List[str]                 = []

    field_map = [
        (cfg.canonical_text_field,    cfg.allowed_text_aliases,    ""),
        (cfg.canonical_start_field,   cfg.allowed_start_aliases,   None),
        (cfg.canonical_end_field,     cfg.allowed_end_aliases,     None),
        (cfg.canonical_regions_field, cfg.allowed_regions_aliases, []),
    ]

    for canonical_name, aliases, default in field_map:
        found_key = _pick_first_existing_key(payload, aliases)
        if found_key is not None:
            normalized[canonical_name] = payload[found_key]
            if found_key != canonical_name:
                changes.append(NormalizationChange(
                    field=canonical_name,
                    action="repair_field_alias",
                    before=found_key,
                    after=canonical_name,
                ))
        else:
            normalized[canonical_name] = default
            if default == "":
                warnings.append(
                    f"Missing transcript/text field; defaulted to empty string."
                )

    # Carry through regionsCount if present (will be recomputed later anyway)
    if "regionsCount" in payload:
        normalized["regionsCount"] = payload["regionsCount"]

    return normalized, changes, warnings


def _normalize_regions(
    regions: Optional[Sequence[Dict[str, Any]]],
    text_config: TextNormalizationConfig,
    payload_config: PayloadNormalizationConfig,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize every region dict in a regions list."""
    if regions is None:
        return [], []
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
        return [], ["Regions field was not a valid list-like sequence; defaulted to []."]

    normalized_regions: List[Dict[str, Any]] = []
    warnings:           List[str]            = []

    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            warnings.append(f"Region {idx} was not an object and was skipped.")
            continue

        region_canonical, _, region_warnings = _canonicalize_payload_keys(
            region, payload_config
        )
        warnings.extend(region_warnings)

        text_result = normalize_text(region_canonical.get("transcript"), text_config)
        start_time  = _normalize_timestamp(region_canonical.get("startTimeSec"))
        end_time    = _normalize_timestamp(region_canonical.get("endTimeSec"))

        if region_canonical.get("transcript") is None:
            warnings.append(f"Region {idx} transcript was null; normalized to empty string.")
        if start_time is None:
            warnings.append(f"Region {idx} has missing or unparseable startTimeSec.")
        if end_time is None:
            warnings.append(f"Region {idx} has missing or unparseable endTimeSec.")

        normalized_regions.append({
            "transcript":   text_result.normalized_text,
            "startTimeSec": start_time,
            "endTimeSec":   end_time,
        })

    return normalized_regions, warnings


def _ensure_regions_count(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[NormalizationChange]]:
    updated = dict(payload)
    before  = updated.get("regionsCount")
    after   = len(updated.get("regions", []))
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
    """Apply fn to text; append a NormalizationChange only if text actually changed."""
    updated = fn(text)
    if updated != text:
        changes.append(NormalizationChange(
            field=field_name,
            action=action,
            before=text,
            after=updated,
        ))
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
        text = (text
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
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


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (python text_normalizer.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Text normalization: source-aware defaults ──────────────────────────
    examples = [
        (TextSource.TASK_ANSWER,   "  I can\u2019t find the Submit button!! \U0001F624"),
        (TextSource.TRANSCRIPTION, "Uh um so I I clicked here and like\u2026 nothing happened you know"),
        (TextSource.STUDY_TEXT,    "Nielsen H3 (User Control) scored poorly. See H7."),
        (TextSource.TASK_ANSWER,   "\u0644\u0627 \u0623\u0633\u062a\u0637\u064a\u0639 \u0627\u0644\u0639\u062b\u0648\u0631"),  # Arabic
        (TextSource.TASK_ANSWER,   "\u9001\u4fe1\u30dc\u30bf\u30f3\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093"),         # Japanese
    ]

    print("=" * 64)
    print("TEXT NORMALIZATION")
    print("=" * 64)
    for source, raw in examples:
        cfg    = get_default_config_for_source(source)
        result = normalize_text(raw, cfg)
        print(f"SOURCE  : {result.source.value}")
        print(f"RAW     : {result.original_text!r}")
        print(f"NORM    : {result.normalized_text!r}")
        print(f"LANG    : {result.language_hint}")
        print(f"CHANGES : {len(result.changes)}")
        print("-" * 64)

    # ── Payload normalization: field aliases + audit trail ─────────────────
    sample_payload = {
        "transcipt": "  Uh... I expected it on the top right!!! ",   # typo alias
        "start": "12.4",                                               # alias
        "end": 18,
        "regions": [
            {"start": "0.0",     "end": "3.1",  "text": "  Hmm, okay... "},
            {"start_sec": 3.1,   "end_sec": 6.5, "utterance": "I can\u2019t find the button???"},
            {"startTimeSec": None, "endTimeSec": 8.2, "transcipt": None},
        ],
        "regionsCount": 99,   # will be recomputed to 3
    }

    print("\n" + "=" * 64)
    print("PAYLOAD NORMALIZATION")
    print("=" * 64)
    result = normalize_transcript_payload(sample_payload)
    print("NORMALIZED PAYLOAD:")
    for k, v in result.normalized_payload.items():
        print(f"  {k}: {v}")
    print(f"\nCHANGES ({len(result.changes)}):")
    for c in result.changes:
        print(f"  [{c.action}] {c.field}: {c.before!r} → {c.after!r}")
    print(f"\nWARNINGS ({len(result.warnings)}):")
    for w in result.warnings:
        print(f"  - {w}")