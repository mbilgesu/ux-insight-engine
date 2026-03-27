from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from text_normalizer import NormalizedRecord, normalize_record

IssueCategory = Literal[
    "navigation",
    "terminology",
    "visual_hierarchy",
    "feedback_missing",
    "form_friction",
    "performance",
    "trust_clarity",
    "error_recovery",
]

SentimentLabel = Literal["negative", "positive", "neutral"]
Severity = Literal["high", "medium", "low"]
TaskSignal = Literal["task_success", "task_failure", "task_delay", "unknown"]


# --- Modular rule config: category -> list of phrases (Layer 1) ---
ISSUE_PHRASES: dict[str, list[str]] = {
    "navigation": [
        "could not find",
        "couldn't find",
        "where is",
        "where's",
        "not obvious",
        "took me a while to find",
        "hard to find",
        "lost",
        "buried in",
    ],
    "terminology": [
        "confusing wording",
        "didn't understand",
        "did not understand",
        "dont understand",
        "unclear label",
        "what does this mean",
        "jargon",
    ],
    "visual_hierarchy": [
        "too small",
        "couldn't see",
        "didn't notice",
        "blends in",
        "hard to spot",
        "missed the",
    ],
    "feedback_missing": [
        "nothing happened",
        "no feedback",
        "didn't know if it worked",
        "did not know if it worked",
        "not sure if it saved",
        "no confirmation",
    ],
    "form_friction": [
        "too many fields",
        "kept re-entering",
        "re-enter",
        "validation issue",
        "invalid field",
        "wouldn't submit",
    ],
    "performance": [
        "slow",
        "lag",
        "laggy",
        "takes too long",
        "loading forever",
        "unresponsive",
    ],
    "trust_clarity": [
        "sketchy",
        "don't trust",
        "is this legit",
        "seems unsafe",
        "unclear pricing",
    ],
    "error_recovery": [
        "error",
        "couldn't fix",
        "stuck",
        "gave up",
        "blocked",
        "crash",
        "broken",
    ],
}

# Sentiment cues (Layer 2) — asymmetric: negative wins over positive
NEGATIVE_SENTIMENT_PHRASES: list[str] = [
    "confusing",
    "confused",
    "frustration",
    "frustrated",
    "frustrating",
    "annoying",
    "doesn't work",
    "didn't work",
    "failed",
    "failure",
    "impossible",
    "waste of time",
    "terrible",
    "hate",
]

POSITIVE_SENTIMENT_PHRASES: list[str] = [
    "easy",
    "clear",
    "worked well",
    "intuitive",
    "straightforward",
    "simple",
    "love",
    "great",
]

# Layer 3 — lexical hints for task outcome (ordered priority for task_signal)
_TASK_FAILURE_PHRASES: list[str] = [
    "gave up",
    "couldn't complete",
    "could not complete",
    "impossible",
    "stuck",
    "blocked",
    "wouldn't let me",
    "error",
    "crash",
    "broken",
]

_TASK_SUCCESS_PHRASES: list[str] = [
    "worked",
    "managed to",
    "finally got it",
    "all good",
    "no problem",
    "easy",
]

_TASK_DELAY_PHRASES: list[str] = [
    "took a while",
    "took me a while",
    "eventually",
    "after several",
    "had to try",
    "slow",
    "lag",
]


@dataclass(frozen=True)
class AnalysisResult:
    record_id: str
    issue_category: IssueCategory | None
    sentiment_label: SentimentLabel
    severity: Severity
    task_signal: TaskSignal
    matched_rules: list[str]
    confidence: float


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    parts = phrase.strip().lower().split()
    if not parts:
        return re.compile(r"a^")  # never matches
    escaped = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


# Precompile issue patterns: (category, phrase, pattern)
_COMPILED_ISSUE_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (cat, phrase, _phrase_regex(phrase))
    for cat, phrases in ISSUE_PHRASES.items()
    for phrase in phrases
]


def _compile_simple_phrases(phrases: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    return [(p, _phrase_regex(p)) for p in phrases]


_COMPILED_NEG = _compile_simple_phrases(NEGATIVE_SENTIMENT_PHRASES)
_COMPILED_POS = _compile_simple_phrases(POSITIVE_SENTIMENT_PHRASES)
_COMPILED_FAIL = _compile_simple_phrases(_TASK_FAILURE_PHRASES)
_COMPILED_OK = _compile_simple_phrases(_TASK_SUCCESS_PHRASES)
_COMPILED_DELAY = _compile_simple_phrases(_TASK_DELAY_PHRASES)


def _find_matches(
    haystack: str, rules: list[tuple[str, re.Pattern[str]]]
) -> list[str]:
    out: list[str] = []
    for label, pat in rules:
        if pat.search(haystack):
            out.append(label)
    return out


def _detect_issue_phrases(haystack: str) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {c: 0 for c in ISSUE_PHRASES}
    matched: list[str] = []
    for category, phrase, pat in _COMPILED_ISSUE_RULES:
        if pat.search(haystack):
            counts[category] += 1
            matched.append(f"{category}:{phrase}")
    return counts, matched


def _infer_category(counts: dict[str, int]) -> IssueCategory | None:
    best = max(counts.values())
    if best == 0:
        return None
    # Deterministic tie-break: lexicographic category name
    winners = sorted([c for c, n in counts.items() if n == best])
    return winners[0]  # type: ignore[return-value]


def _infer_sentiment(haystack: str) -> tuple[SentimentLabel, list[str]]:
    neg_hits = _find_matches(haystack, _COMPILED_NEG)
    pos_hits = _find_matches(haystack, _COMPILED_POS)
    rules: list[str] = []
    if neg_hits:
        rules.extend(f"sentiment_negative:{h}" for h in neg_hits)
    if pos_hits:
        rules.extend(f"sentiment_positive:{h}" for h in pos_hits)
    if neg_hits:
        return "negative", rules
    if pos_hits:
        return "positive", rules
    return "neutral", rules


def _infer_task_signal(haystack: str) -> tuple[TaskSignal, list[str]]:
    fail_hits = _find_matches(haystack, _COMPILED_FAIL)
    if fail_hits:
        return "task_failure", [f"task_failure:{h}" for h in fail_hits]
    ok_hits = _find_matches(haystack, _COMPILED_OK)
    if ok_hits:
        return "task_success", [f"task_success:{h}" for h in ok_hits]
    delay_hits = _find_matches(haystack, _COMPILED_DELAY)
    if delay_hits:
        return "task_delay", [f"task_delay:{h}" for h in delay_hits]
    return "unknown", []


def _infer_severity(
    task_signal: TaskSignal,
    sentiment: SentimentLabel,
    category: IssueCategory | None,
    haystack: str,
    issue_matched: list[str],
) -> tuple[Severity, list[str]]:
    extra: list[str] = []
    # High: blocked / failed flow
    if task_signal == "task_failure":
        extra.append("severity_rule:task_failure->high")
        return "high", extra
    if re.search(r"\bstuck\b|\bblocked\b|\bgave up\b|\bcouldn'?t complete\b", haystack):
        extra.append("severity_rule:blocked_lexicon->high")
        return "high", extra
    # Medium: confusion / friction with evidence, or delay with negative sentiment
    if task_signal == "task_delay" and sentiment == "negative":
        extra.append("severity_rule:delay+negative->medium")
        return "medium", extra
    if category is not None and issue_matched:
        if sentiment == "negative" or task_signal == "task_delay":
            extra.append("severity_rule:category_evidence+strain->medium")
            return "medium", extra
        extra.append("severity_rule:category_evidence->low")
        return "low", extra
    if sentiment == "negative":
        extra.append("severity_rule:negative_only->medium")
        return "medium", extra
    extra.append("severity_rule:default->low")
    return "low", extra


def _confidence(
    haystack: str,
    category: IssueCategory | None,
    counts: dict[str, int],
    matched_rules: list[str],
) -> float:
    if not haystack:
        return 0.0
    n_chars = max(len(haystack), 1)
    evidence = len([r for r in matched_rules if not r.startswith("sentiment_")])
    signal_density = min(evidence * 6 / n_chars, 0.55)
    total_hits = sum(counts.values())
    if category is None:
        coherence = 0.0
    else:
        share = counts.get(category, 0) / max(total_hits, 1)
        coherence = 0.25 + 0.45 * share
    return round(min(signal_density + coherence, 1.0), 3)


def analyze_record(record: NormalizedRecord) -> AnalysisResult:
    haystack = _collapse_ws(record.text)
    counts, issue_rules = _detect_issue_phrases(haystack)
    category = _infer_category(counts)

    sentiment, sent_rules = _infer_sentiment(haystack)
    task_signal, task_rules = _infer_task_signal(haystack)
    severity, sev_rules = _infer_severity(
        task_signal, sentiment, category, haystack, issue_rules
    )

    matched_rules = issue_rules + sent_rules + task_rules + sev_rules
    conf = _confidence(haystack, category, counts, matched_rules)

    return AnalysisResult(
        record_id=record.record_id,
        issue_category=category,
        sentiment_label=sentiment,
        severity=severity,
        task_signal=task_signal,
        matched_rules=matched_rules,
        confidence=conf,
    )


def analyze_records(records: list[NormalizedRecord]) -> list[AnalysisResult]:
    return [analyze_record(r) for r in records]


def normalize_and_analyze(
    record_id: str,
    raw_text: str,
    source: str = "unknown",
) -> tuple[NormalizedRecord, AnalysisResult]:
    """
    Full pipeline: RUXAILAB-style normalization → semantic signal extraction.
    Returns the normalized record (with metadata) and the analysis result.
    """
    rec = normalize_record(record_id, raw_text, source)
    return rec, analyze_record(rec)


def analyze_text(
    record_id: str,
    raw_text: str,
    source: str = "unknown",
) -> AnalysisResult:
    """Normalize raw text then analyze; use normalize_and_analyze if you need metadata."""
    _, out = normalize_and_analyze(record_id, raw_text, source)
    return out
