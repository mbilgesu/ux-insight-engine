"""
ruxailab_nlp.py
===============
NLP preprocessing + usability methodology analysis for RUXAILAB.

**Module 1 — text normalization** is implemented in ``text_normalizer.py`` (URL/emoji strip,
payload repair, source-aware cleaning). It is re-exported here so callers can use a single
import path.

**Modules 2–4** (Nielsen mapper, SUS, NASA-TLX) live in ``ruxailab_methodology.py``.

See module docstrings in those files for Firestore-oriented data notes.
"""

from __future__ import annotations

from app.services import ruxailab_methodology as _m
from app.services.text_normalizer import (
    NormalizationChange,
    NormalizedRecord,
    PayloadNormalizationConfig,
    PayloadNormalizationResult,
    TextNormalizationConfig,
    TextNormalizationResult,
    TextSource,
    get_default_config_for_source,
    normalize_record,
    normalize_text,
    normalize_transcript_payload,
)

__all__ = [
    *[
        "TextSource",
        "TextNormalizationConfig",
        "PayloadNormalizationConfig",
        "NormalizationChange",
        "TextNormalizationResult",
        "PayloadNormalizationResult",
        "NormalizedRecord",
        "normalize_text",
        "normalize_transcript_payload",
        "normalize_record",
        "get_default_config_for_source",
    ],
    *_m.__all__,
]

for _name in _m.__all__:
    globals()[_name] = getattr(_m, _name)
