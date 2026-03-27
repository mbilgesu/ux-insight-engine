"""
explainability.py
=================
Explainability layer for RUXAILAB's NLP/usability analysis pipeline.

Two responsibilities:

  1. SEMANTIC ANALYSIS  (SemanticAnalyzer)
     Classifies free-text task answers and transcription segments into
     usability issue categories that match semantic_analysis.py exactly:

       navigation · terminology · visual_hierarchy · feedback_missing
       form_friction · performance · trust_clarity · error_recovery

     Every classification decision is recorded as a MatchedRule so nothing
     is a black box.

     Input  : normalized text (from ``text_normalizer.normalize_text`` / ``normalize_record``)
             + optional TaskContext (task completion, time, source field)
     Output : AnalysisResult — category, severity, sentiment, confidence,
                                and the full matched_rules audit trail

  2. EXPLAINABILITY ENGINE  (ExplainabilityEngine)
     Wraps AnalysisResult in a human-readable explanation that traces every
     decision step, cites the Nielsen heuristic, and provides an actionable
     design recommendation.

     Input  : AnalysisResult
     Output : ExplainedResult — evidence, reasoning_chain, nielsen_ref,
                                  recommendation

WHY THIS IS XAI (EXPLAINABLE AI), NOT JUST AI
──────────────────────────────────────────────
Most NLP pipelines produce a label.  This module produces a reasoning_chain:

  "Step 1 [Input]       Analyzed: 'i could not find the submit button'"
  "Step 2 [Negation]    Negation scope: 'could not' → nav rule escalated"
  "Step 3 [Pattern]     navigation (rule: nav_negated_find, +0.50 conf)"
  "Step 4 [Context]     task_completed=False → severity escalated to HIGH"
  "Step 5 [Heuristic]   navigation → Nielsen H6: Recognition over Recall"
  "Step 6 [Recommendation] Add persistent breadcrumbs and visible CTAs…"

Every step is auditable, reproducible, and free of external API calls.

INTEGRATION WITH RUXAILAB DATA MODEL
──────────────────────────────────────
  TaskAnswer.taskAnswer        → primary text input for SemanticAnalyzer
  TaskAnswer.taskObservations  → secondary text (moderator notes)
  TaskAnswer.completed         → TaskContext.task_completed
  TaskAnswer.taskTime          → TaskContext.task_time_seconds
  TaskAnswer.taskId            → TaskContext.task_id
  Study.studyConclusion        → TextSource.STUDY_TEXT config
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ruxailab_methodology import HeuristicMatch, NielsenMapper

__all__ = [
    "IssueSeverity",
    "IssueSentiment",
    "IssueCategory",
    "TaskContext",
    "MatchedRule",
    "AnalysisResult",
    "SemanticAnalyzer",
    "ExplainedResult",
    "ExplainabilityEngine",
    "StudyExplainabilityReport",
    "StudyExplainabilityAggregator",
    "analyze_and_explain",
]


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════


class IssueSeverity(str, Enum):
    """
    Three-tier scale matching RUXAILAB's accessibility severity colors.
    HIGH → "error" chip | MEDIUM → "warning" | LOW → "success"
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class IssueSentiment(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class IssueCategory(str, Enum):
    """
    Usability issue categories.

    The 8 core values are intentionally aligned with semantic_analysis.py
    so results from both modules can be compared, merged, and routed
    through the same Nielsen mapper and recommendation matrix without
    any translation step.

    POSITIVE_SIGNAL and UNKNOWN are pipeline-level sentinels — they do not
    represent usability violations.
    """

    # ── Core 8 — must match semantic_analysis.py exactly ──────────────────
    NAVIGATION = "navigation"
    TERMINOLOGY = "terminology"
    VISUAL_HIERARCHY = "visual_hierarchy"
    FEEDBACK_MISSING = "feedback_missing"
    FORM_FRICTION = "form_friction"
    PERFORMANCE = "performance"
    TRUST_CLARITY = "trust_clarity"
    ERROR_RECOVERY = "error_recovery"
    # ── Pipeline sentinels ─────────────────────────────────────────────────
    POSITIVE_SIGNAL = "positive_signal"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TaskContext:
    """
    Signals from TaskAnswer fields that influence severity classification.
    All fields optional — degrades gracefully when context is unavailable.
    """

    task_id: Optional[str] = None
    task_name: Optional[str] = None
    task_completed: Optional[bool] = None  # TaskAnswer.completed
    task_attempted: bool = True
    task_time_seconds: Optional[float] = None  # TaskAnswer.taskTime (parsed)
    expected_time_seconds: Optional[float] = None  # Task.estimatedTime
    source_field: str = "taskAnswer"
    study_type: Optional[str] = None  # "HEURISTIC" | "USER"
    respondent_id: Optional[str] = None


@dataclass
class MatchedRule:
    """
    Atomic record of a single pattern that fired during semantic analysis.

    This is the fundamental unit of explainability — every classification
    decision traces back to one or more MatchedRule instances.
    """

    rule_id: str
    category: IssueCategory
    matched_text: str
    pattern_description: str
    confidence_contribution: float
    severity_signal: Optional[IssueSeverity] = None
    sentiment_signal: Optional[IssueSentiment] = None
    negation_detected: bool = False
    position_in_text: Optional[int] = None


@dataclass
class AnalysisResult:
    """
    Output of SemanticAnalyzer for a single text input.

    CONTRACT: every field is fully determined by matched_rules + task_context.
    No hidden state; the reasoning_chain can always be reconstructed.
    """

    raw_text: str
    issue_category: IssueCategory
    secondary_category: Optional[IssueCategory]
    severity: IssueSeverity
    sentiment: IssueSentiment
    confidence: float
    matched_rules: list[MatchedRule]
    task_context: Optional[TaskContext]
    has_negation: bool
    word_count: int

    def is_usability_issue(self) -> bool:
        return self.issue_category not in (IssueCategory.POSITIVE_SIGNAL, IssueCategory.UNKNOWN)

    def as_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "issue_category": self.issue_category.value,
            "secondary_category": self.secondary_category.value if self.secondary_category else None,
            "severity": self.severity.value,
            "sentiment": self.sentiment.value,
            "confidence": round(self.confidence, 3),
            "has_negation": self.has_negation,
            "word_count": self.word_count,
            "matched_rules_count": len(self.matched_rules),
        }


@dataclass
class ExplainedResult:
    """
    Human-readable explanation wrapping an AnalysisResult.

    reasoning_chain is designed to be pasted verbatim into
    FinalReportView's studyConclusion field.
    """

    analysis: AnalysisResult
    evidence: list[str]
    reasoning_chain: list[str]
    nielsen_ref: Optional[str]
    nielsen_description: Optional[str]
    recommendation: str
    secondary_nielsen_ref: Optional[str] = None
    confidence_explanation: str = ""
    is_high_priority: bool = False

    def summary(self) -> str:
        cat = self.analysis.issue_category.value.replace("_", " ").title()
        sev = self.analysis.severity.value
        nref = f" [{self.nielsen_ref}]" if self.nielsen_ref else ""
        return f"[{sev}] {cat}{nref} — {self.recommendation}"

    def reasoning_chain_text(self) -> str:
        return "\n".join(self.reasoning_chain)

    def as_dict(self) -> dict:
        return {
            **self.analysis.as_dict(),
            "evidence": self.evidence,
            "reasoning_chain": self.reasoning_chain,
            "nielsen_ref": self.nielsen_ref,
            "nielsen_description": self.nielsen_description,
            "recommendation": self.recommendation,
            "secondary_nielsen_ref": self.secondary_nielsen_ref,
            "confidence_explanation": self.confidence_explanation,
            "is_high_priority": self.is_high_priority,
            "summary": self.summary(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC ANALYSIS RULES
# ══════════════════════════════════════════════════════════════════════════════

_NEGATION_WINDOW = 5

_NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "nothing",
        "cannot",
        "can not",
        "couldn't",
        "couldn",
        "couldn not",
        "didn't",
        "didn",
        "didn not",
        "don't",
        "don",
        "don not",
        "wasn't",
        "wasn",
        "wasn not",
        "won't",
        "won",
        "won not",
        "isn't",
        "isn",
        "isn not",
        "unable",
        "fail",
        "failed",
        "without",
        "lack",
        "lacking",
        "missing",
    }
)

# Rule format:
# (rule_id, IssueCategory, pattern_str, description, confidence, severity_signal, sentiment_signal)

_RAW_RULES: list[tuple] = [
    # ── navigation ────────────────────────────────────────────────────────────
    (
        "nav_find",
        IssueCategory.NAVIGATION,
        r"\b(find|found|locate|spot|see)\b",
        "Navigation-related 'find/locate' verb",
        0.25,
        None,
        None,
    ),
    (
        "nav_lost",
        IssueCategory.NAVIGATION,
        r"\b(lost|confused|don.t know where|where (is|was|are)|no idea)\b",
        "User expressed being lost or disoriented",
        0.45,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "nav_back",
        IssueCategory.NAVIGATION,
        r"\b(back\s+button|go\s+back|return|previous\s+page|breadcrumb)\b",
        "Back/return navigation mentioned",
        0.30,
        IssueSeverity.MEDIUM,
        None,
    ),
    (
        "nav_expected",
        IssueCategory.NAVIGATION,
        r"\b(expected (it|this|that) (to be|on|at|in)|thought it was|should be (on|at|in))\b",
        "User expressed expectation mismatch about element location",
        0.40,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "nav_click_nothing",
        IssueCategory.NAVIGATION,
        r"\b(click(ed)?|tap(ped)?|press(ed)?)\b.{0,30}\b(nothing|nothing happened|no response|didn.t work)\b",
        "Click/tap with no visible result",
        0.55,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    # ── feedback_missing ──────────────────────────────────────────────────────
    (
        "fb_loading",
        IssueCategory.FEEDBACK_MISSING,
        r"\b(load(ing|ed)?|wait(ing|ed)?|forever|tak(e|ing|es) (long|too long|a while|forever))\b",
        "Loading / response time issue indicating missing feedback",
        0.35,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "fb_no_feedback",
        IssueCategory.FEEDBACK_MISSING,
        r"\b(no (response|feedback|confirmation|message)|nothing (happened|showed|appeared)|didn.t (show|appear|respond|update))\b",
        "Absence of system feedback after user action",
        0.55,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "fb_spinner",
        IssueCategory.FEEDBACK_MISSING,
        r"\b(spinner|progress|indicator|bar|status)\b",
        "Progress/status indicator mentioned as missing or unclear",
        0.20,
        None,
        None,
    ),
    # ── performance ───────────────────────────────────────────────────────────
    (
        "perf_slow",
        IssueCategory.PERFORMANCE,
        r"\b(slow|lag(g(ing|ed))?|sluggish|unresponsive)\b",
        "Slowness or lag detected",
        0.50,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "perf_timeout",
        IssueCategory.PERFORMANCE,
        r"\b(timeout|timed\s+out|took\s+too\s+long|takes\s+forever|still\s+loading)\b",
        "Timeout or excessive load time",
        0.60,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "perf_freeze",
        IssueCategory.PERFORMANCE,
        r"\b(freez(e|ing|es)|hang(ing)?|stuck\s+on\s+loading|not\s+responding)\b",
        "Application freeze or hang",
        0.65,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    # ── error_recovery ────────────────────────────────────────────────────────
    (
        "err_crash",
        IssueCategory.ERROR_RECOVERY,
        r"\b(crash(ed)?|broke|broken|stop(ped)?\s+working|error\s+page|white\s+(screen|page))\b",
        "Application crash or fatal failure",
        0.70,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "err_message",
        IssueCategory.ERROR_RECOVERY,
        r"\b(error\s+message|warning\s+message|alert|notification)\b",
        "Error/warning message mentioned",
        0.30,
        IssueSeverity.MEDIUM,
        None,
    ),
    (
        "err_unclear_msg",
        IssueCategory.ERROR_RECOVERY,
        r"\b(didn.t (understand|know what|get)|what does.{0,20}mean|unclear|confusing (message|error|warning))\b",
        "Error message was unclear or incomprehensible",
        0.50,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "err_undo",
        IssueCategory.ERROR_RECOVERY,
        r"\b(undo|redo|go\s+back|revert|cancel|reverse)\b",
        "User sought undo/redo/cancel functionality",
        0.35,
        IssueSeverity.MEDIUM,
        None,
    ),
    (
        "err_deleted",
        IssueCategory.ERROR_RECOVERY,
        r"\b(deleted|removed|lost\s+my|gone|disappeared|accidentally)\b",
        "Data loss or accidental destructive action",
        0.65,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    # ── terminology ───────────────────────────────────────────────────────────
    (
        "term_jargon",
        IssueCategory.TERMINOLOGY,
        r"\b(jargon|technical|abbreviation|acronym|what (does|is)\s+\w+\s+(mean|stand for))\b",
        "Technical jargon or unexplained abbreviation detected",
        0.50,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "term_confusing_label",
        IssueCategory.TERMINOLOGY,
        r"\b(label|button|link|menu)\b.{0,20}\b(confusing|unclear|strange|odd|weird|doesn.t make sense)\b",
        "UI element label described as confusing",
        0.55,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "term_wrong_word",
        IssueCategory.TERMINOLOGY,
        r"\b(wrong word|bad label|misleading|mislabeled|misnamed)\b",
        "Incorrect or misleading terminology identified",
        0.60,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    # ── visual_hierarchy ──────────────────────────────────────────────────────
    (
        "vis_clutter",
        IssueCategory.VISUAL_HIERARCHY,
        r"\b(clutter(ed)?|messy|overwhelm(ing|ed)?|too\s+much\s+(information|text|stuff|content)|busy)\b",
        "Information overload / visual clutter",
        0.45,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "vis_small",
        IssueCategory.VISUAL_HIERARCHY,
        r"\b(too\s+small|tiny|hard\s+to\s+(see|read|tap|click)|small\s+text|small\s+button)\b",
        "Elements too small to interact with comfortably",
        0.45,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "vis_hidden",
        IssueCategory.VISUAL_HIERARCHY,
        r"\b(hidden|buried|hard\s+to\s+find|tucked\s+away|not\s+visible|invisible|missed\s+it)\b",
        "Important element was not visible enough",
        0.50,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    # ── form_friction ─────────────────────────────────────────────────────────
    (
        "form_validation",
        IssueCategory.FORM_FRICTION,
        r"\b(required\s+field|mandatory|must\s+(fill|enter|provide)|missing\s+field|form\s+error)\b",
        "Form validation issue",
        0.45,
        IssueSeverity.MEDIUM,
        None,
    ),
    (
        "form_lost_data",
        IssueCategory.FORM_FRICTION,
        r"\b(lost\s+(my\s+)?data|cleared|reset\s+the\s+form|had\s+to\s+(re.?enter|retype|redo))\b",
        "Form data was lost and had to be re-entered",
        0.65,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "form_too_many",
        IssueCategory.FORM_FRICTION,
        r"\b(too\s+many\s+(fields|inputs|questions)|long\s+form|form\s+is\s+long)\b",
        "Form has too many fields causing friction",
        0.50,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    # ── trust_clarity ─────────────────────────────────────────────────────────
    (
        "trust_unsafe",
        IssueCategory.TRUST_CLARITY,
        r"\b(not\s+sure\s+if\s+(safe|secure)|looks\s+sketchy|don.t\s+trust|feels\s+unofficial)\b",
        "User expressed safety or trust concern",
        0.60,
        IssueSeverity.HIGH,
        IssueSentiment.NEGATIVE,
    ),
    (
        "trust_privacy",
        IssueCategory.TRUST_CLARITY,
        r"\b(privacy|data\s+(sharing|collection)|who\s+can\s+see|personal\s+information)\b",
        "Privacy or data concern mentioned",
        0.50,
        IssueSeverity.MEDIUM,
        IssueSentiment.NEGATIVE,
    ),
    (
        "trust_secure",
        IssueCategory.TRUST_CLARITY,
        r"\b(is\s+this\s+secure|https|padlock|ssl|certificate)\b",
        "Security indicator question or concern",
        0.45,
        IssueSeverity.MEDIUM,
        None,
    ),
    # ── positive_signal ───────────────────────────────────────────────────────
    (
        "pos_easy",
        IssueCategory.POSITIVE_SIGNAL,
        r"\b(easy|intuitive|clear|simple|obvious|straightforward|smooth|great|love(d)?|excellent|perfect)\b",
        "Positive usability signal",
        0.40,
        IssueSeverity.NONE,
        IssueSentiment.POSITIVE,
    ),
    (
        "pos_found",
        IssueCategory.POSITIVE_SIGNAL,
        r"\b(found\s+it\s+(easily|quickly|right away)|no\s+problem|worked\s+(well|great|fine))\b",
        "User completed task without difficulty",
        0.50,
        IssueSeverity.NONE,
        IssueSentiment.POSITIVE,
    ),
]


@dataclass
class _CompiledRule:
    rule_id: str
    category: IssueCategory
    pattern: re.Pattern
    description: str
    confidence: float
    severity_signal: Optional[IssueSeverity]
    sentiment_signal: Optional[IssueSentiment]


_COMPILED_RULES: list[_CompiledRule] = [
    _CompiledRule(
        rule_id=rid,
        category=cat,
        pattern=re.compile(pat, re.IGNORECASE | re.DOTALL),
        description=desc,
        confidence=conf,
        severity_signal=sev_sig,
        sentiment_signal=sent_sig,
    )
    for rid, cat, pat, desc, conf, sev_sig, sent_sig in _RAW_RULES
]

_NEGATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_NEGATION_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION MATRIX
# (IssueCategory, IssueSeverity) → actionable recommendation
# ══════════════════════════════════════════════════════════════════════════════

_RECOMMENDATIONS: dict[tuple[IssueCategory, IssueSeverity], str] = {
    (
        IssueCategory.NAVIGATION,
        IssueSeverity.HIGH,
    ): (
        "Add persistent breadcrumbs and clearly visible primary CTAs on every page. "
        "Consider a fixed navigation sidebar so users always know where they are."
    ),
    (
        IssueCategory.NAVIGATION,
        IssueSeverity.MEDIUM,
    ): (
        "Review information architecture: label navigation items with user-familiar terms "
        "and ensure the active page is always visually highlighted."
    ),
    (
        IssueCategory.NAVIGATION,
        IssueSeverity.LOW,
    ): (
        "Consider adding contextual 'back' links or a mini-sitemap at the bottom of deep pages."
    ),
    (
        IssueCategory.TERMINOLOGY,
        IssueSeverity.HIGH,
    ): (
        "Replace all technical terminology and unexplained acronyms with plain-language "
        "equivalents. Run a terminology audit with representative users."
    ),
    (
        IssueCategory.TERMINOLOGY,
        IssueSeverity.MEDIUM,
    ): (
        "Add tooltips to technical terms and ensure every button/link label uses a "
        "verb-noun pattern that matches user expectations."
    ),
    (
        IssueCategory.TERMINOLOGY,
        IssueSeverity.LOW,
    ): (
        "Review copy with a plain-language lens. Aim for 8th-grade reading level "
        "in all UI labels and help text."
    ),
    (
        IssueCategory.VISUAL_HIERARCHY,
        IssueSeverity.HIGH,
    ): (
        "Reduce information density: move secondary content to a collapsible panel. "
        "Increase touch targets to ≥ 44×44px and apply a clear visual hierarchy."
    ),
    (
        IssueCategory.VISUAL_HIERARCHY,
        IssueSeverity.MEDIUM,
    ): (
        "Apply visual hierarchy: limit each screen to one primary action. "
        "Use whitespace and typographic weight to guide the eye to key elements."
    ),
    (
        IssueCategory.VISUAL_HIERARCHY,
        IssueSeverity.LOW,
    ): (
        "Review alignment and spacing using an 8pt grid. "
        "Ensure interactive elements have sufficient touch targets and contrast ratios."
    ),
    (
        IssueCategory.FEEDBACK_MISSING,
        IssueSeverity.HIGH,
    ): (
        "Add immediate visual confirmation (spinner, toast, or progress bar) for every "
        "user action that takes more than 300ms. Users must never wonder if their action registered."
    ),
    (
        IssueCategory.FEEDBACK_MISSING,
        IssueSeverity.MEDIUM,
    ): (
        "Audit all async operations and add status indicators. "
        "Even a simple 'Loading…' message prevents abandonment."
    ),
    (
        IssueCategory.FEEDBACK_MISSING,
        IssueSeverity.LOW,
    ): (
        "Improve micro-interactions: add subtle animations to confirm state changes "
        "(button press, form submission, filter application)."
    ),
    (
        IssueCategory.FORM_FRICTION,
        IssueSeverity.HIGH,
    ): (
        "Implement inline validation with real-time feedback. "
        "Never clear form data on validation error. "
        "Show field-level error messages adjacent to the offending input."
    ),
    (
        IssueCategory.FORM_FRICTION,
        IssueSeverity.MEDIUM,
    ): (
        "Add format hints (placeholder text, input masks) to prevent errors before they occur. "
        "Preserve all field values on page reload and error states."
    ),
    (
        IssueCategory.FORM_FRICTION,
        IssueSeverity.LOW,
    ): (
        "Consider auto-saving draft form state. "
        "Add smart defaults and auto-fill where appropriate."
    ),
    (
        IssueCategory.PERFORMANCE,
        IssueSeverity.HIGH,
    ): (
        "Profile and fix the slowest operations immediately. "
        "Add loading skeletons or progress bars so users know the system is working. "
        "Target < 3s for all primary task interactions."
    ),
    (
        IssueCategory.PERFORMANCE,
        IssueSeverity.MEDIUM,
    ): (
        "Lazy-load non-critical content and optimise the critical rendering path. "
        "Add a visible spinner for any operation exceeding 1s."
    ),
    (
        IssueCategory.PERFORMANCE,
        IssueSeverity.LOW,
    ): (
        "Review asset sizes and consider CDN caching for static resources. "
        "Use optimistic UI updates to mask latency for common actions."
    ),
    (
        IssueCategory.TRUST_CLARITY,
        IssueSeverity.HIGH,
    ): (
        "Add visible trust signals: HTTPS padlock, privacy policy link, and clear data-use "
        "statements at the point of data collection. Conduct a security review."
    ),
    (
        IssueCategory.TRUST_CLARITY,
        IssueSeverity.MEDIUM,
    ): (
        "Clarify data handling in plain language near sensitive inputs. "
        "Add a brief 'Why we ask for this' tooltip to each personal data field."
    ),
    (
        IssueCategory.TRUST_CLARITY,
        IssueSeverity.LOW,
    ): (
        "Ensure the domain, brand identity, and visual design consistently signal "
        "legitimacy. Remove any design elements that could appear unofficial."
    ),
    (
        IssueCategory.ERROR_RECOVERY,
        IssueSeverity.HIGH,
    ): (
        "Rewrite error messages in plain language with a specific cause and a concrete "
        "next step. Add an Undo action for all destructive operations."
    ),
    (
        IssueCategory.ERROR_RECOVERY,
        IssueSeverity.MEDIUM,
    ): (
        "Audit error message copy: each message should state what went wrong and "
        "what the user should do next. Remove all error codes from user-facing text."
    ),
    (
        IssueCategory.ERROR_RECOVERY,
        IssueSeverity.LOW,
    ): (
        "Add confirmation dialogs before irreversible actions "
        "(delete, submit, overwrite) to prevent accidental errors."
    ),
    (
        IssueCategory.POSITIVE_SIGNAL,
        IssueSeverity.NONE,
    ): (
        "Maintain this design pattern — it is working well for users. "
        "Document it in your design system as a reference implementation."
    ),
    (
        IssueCategory.UNKNOWN,
        IssueSeverity.NONE,
    ): (
        "Review the raw response manually — the automated classifier could not "
        "determine a usability issue category with sufficient confidence."
    ),
}


def _get_recommendation(category: IssueCategory, severity: IssueSeverity) -> str:
    return _RECOMMENDATIONS.get(
        (category, severity),
        _RECOMMENDATIONS.get(
            (category, IssueSeverity.MEDIUM),
            "Review this finding with the design team and prioritise based on task impact.",
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC ANALYZER
# ══════════════════════════════════════════════════════════════════════════════


class SemanticAnalyzer:
    """
    Classifies normalized task-answer text into usability issue categories.

    Every classification is backed by a list of MatchedRule instances that
    form the audit trail for the explainability layer.

    Quick usage
    -----------
    >>> analyzer = SemanticAnalyzer()
    >>> result = analyzer.analyze(
    ...     "i could not find the submit button",
    ...     context=TaskContext(task_completed=False)
    ... )
    >>> result.issue_category
    <IssueCategory.NAVIGATION: 'navigation'>
    >>> result.severity
    <IssueSeverity.HIGH: 'HIGH'>
    """

    def analyze(self, text: str, context: Optional[TaskContext] = None) -> AnalysisResult:
        text = text.strip() if text else ""
        word_count = len(text.split()) if text else 0

        # ── 1. Negation pass ──────────────────────────────────────────────────
        negation_positions = {m.start() for m in _NEGATION_RE.finditer(text)}

        # ── 2. Rule matching ──────────────────────────────────────────────────
        fired_rules: list[MatchedRule] = []
        for rule in _COMPILED_RULES:
            for match in rule.pattern.finditer(text):
                negated = _is_negated(match.start(), text, negation_positions)
                effective_cat = rule.category

                # Negated positive signal → flip to navigation (most common culprit)
                if negated and rule.category == IssueCategory.POSITIVE_SIGNAL:
                    effective_cat = IssueCategory.NAVIGATION

                # Negation boosts confidence and escalates severity
                neg_boost = 0.25 if negated and rule.category != IssueCategory.POSITIVE_SIGNAL else 0.0

                fired_rules.append(
                    MatchedRule(
                        rule_id=rule.rule_id,
                        category=effective_cat,
                        matched_text=match.group(0),
                        pattern_description=rule.description,
                        confidence_contribution=min(1.0, rule.confidence + neg_boost),
                        severity_signal=(
                            IssueSeverity.HIGH
                            if negated and rule.severity_signal == IssueSeverity.MEDIUM
                            else rule.severity_signal
                        ),
                        sentiment_signal=(
                            IssueSentiment.NEGATIVE
                            if negated and rule.sentiment_signal in (None, IssueSentiment.POSITIVE)
                            else rule.sentiment_signal
                        ),
                        negation_detected=negated,
                        position_in_text=match.start(),
                    )
                )

        has_negation = any(r.negation_detected for r in fired_rules)

        # ── 3–6. Resolve category, confidence, severity, sentiment ────────────
        issue_cat, secondary_cat = _determine_categories(fired_rules)
        confidence = _compute_confidence(fired_rules, issue_cat)
        severity = _determine_severity(fired_rules, issue_cat, context, confidence)
        sentiment = _determine_sentiment(fired_rules, issue_cat)

        return AnalysisResult(
            raw_text=text,
            issue_category=issue_cat,
            secondary_category=secondary_cat,
            severity=severity,
            sentiment=sentiment,
            confidence=confidence,
            matched_rules=fired_rules,
            task_context=context,
            has_negation=has_negation,
            word_count=word_count,
        )

    def analyze_batch(
        self,
        texts: list[str],
        contexts: Optional[list[Optional[TaskContext]]] = None,
    ) -> list[AnalysisResult]:
        contexts = contexts or [None] * len(texts)
        return [self.analyze(t, c) for t, c in zip(texts, contexts)]


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Bridge: IssueCategory → keyword for NielsenMapper (Tier-2 free-text path)
_CATEGORY_TO_KEYWORD: dict[IssueCategory, str] = {
    IssueCategory.NAVIGATION: "navigation",
    IssueCategory.TERMINOLOGY: "terminology",
    IssueCategory.VISUAL_HIERARCHY: "visual hierarchy cluttered interface",
    IssueCategory.FEEDBACK_MISSING: "feedback missing no response",
    IssueCategory.FORM_FRICTION: "form validation friction",
    IssueCategory.PERFORMANCE: "loading slow performance",
    IssueCategory.TRUST_CLARITY: "trust security privacy",
    IssueCategory.ERROR_RECOVERY: "error recovery undo",
    IssueCategory.POSITIVE_SIGNAL: "easy intuitive",
    IssueCategory.UNKNOWN: "unknown",
}


class ExplainabilityEngine:
    """
    Wraps AnalysisResult in a fully traceable explanation.

    The engine never re-runs analysis — it receives a completed AnalysisResult
    and reconstructs the logical chain that produced it.
    """

    def explain(self, result: AnalysisResult) -> ExplainedResult:
        evidence = self._build_evidence(result)
        reasoning_chain = self._build_reasoning_chain(result)
        hm = self._map_to_heuristic(result.issue_category)
        nielsen_ref = self._format_ref(hm, primary=True)
        nielsen_desc = self._format_desc(hm)
        secondary_ref = self._format_ref(hm, primary=False)
        recommendation = _get_recommendation(result.issue_category, result.severity)
        conf_expl = self._confidence_explanation(result)
        is_high_priority = result.severity == IssueSeverity.HIGH and result.confidence >= 0.6
        return ExplainedResult(
            analysis=result,
            evidence=evidence,
            reasoning_chain=reasoning_chain,
            nielsen_ref=nielsen_ref,
            nielsen_description=nielsen_desc,
            recommendation=recommendation,
            secondary_nielsen_ref=secondary_ref,
            confidence_explanation=conf_expl,
            is_high_priority=is_high_priority,
        )

    def explain_batch(self, results: list[AnalysisResult]) -> list[ExplainedResult]:
        return [self.explain(r) for r in results]

    def _build_evidence(self, result: AnalysisResult) -> list[str]:
        ev: list[str] = []
        seen: set[str] = set()

        for rule in result.matched_rules:
            if rule.matched_text in seen:
                continue
            seen.add(rule.matched_text)
            neg_prefix = "Negated " if rule.negation_detected else ""
            sev_suffix = (
                f" (severity signal: {rule.severity_signal.value})" if rule.severity_signal else ""
            )
            ev.append(f"{neg_prefix}Found '{rule.matched_text}': {rule.pattern_description}{sev_suffix}")

        ctx = result.task_context
        if ctx:
            if ctx.task_completed is False:
                ev.append("Context: task_completed=False — task failure confirmed by study data")
            if ctx.task_completed is True:
                ev.append("Context: task_completed=True — issue noted but task was completed")
            if (
                ctx.task_time_seconds
                and ctx.expected_time_seconds
                and ctx.task_time_seconds > ctx.expected_time_seconds * 1.5
            ):
                ev.append(
                    f"Context: task took {ctx.task_time_seconds:.0f}s "
                    f"(expected ≤ {ctx.expected_time_seconds:.0f}s) — significant time overrun"
                )
            if ctx.source_field == "taskObservations":
                ev.append(
                    "Source: moderator observation notes — "
                    "reflects researcher judgement, not participant self-report"
                )

        if not ev:
            ev.append("No specific evidence patterns matched — classification is low-confidence.")
        return ev

    def _build_reasoning_chain(self, result: AnalysisResult) -> list[str]:
        chain: list[str] = []
        step = 1

        preview = (result.raw_text[:80] + "…") if len(result.raw_text) > 80 else result.raw_text
        chain.append(f"Step {step} [Input]        Analyzed text ({result.word_count} words): '{preview}'")
        step += 1

        negated_rules = [r for r in result.matched_rules if r.negation_detected]
        if negated_rules:
            examples = ", ".join(f"'{r.matched_text}'" for r in negated_rules[:3])
            chain.append(
                f"Step {step} [Negation]     Negation scope detected around: {examples}. "
                f"Confidence and severity escalated for affected rules."
            )
            step += 1

        primary_rules = [r for r in result.matched_rules if r.category == result.issue_category]
        if primary_rules:
            top = primary_rules[:3]
            summary = "; ".join(
                f"rule '{r.rule_id}' matched '{r.matched_text}' (+{r.confidence_contribution:.2f})"
                for r in top
            )
            chain.append(
                f"Step {step} [Pattern]      Category '{result.issue_category.value}' "
                f"signaled by: {summary}."
            )
        else:
            chain.append(
                f"Step {step} [Pattern]      No primary patterns matched. "
                f"Category '{result.issue_category.value}' assigned as fallback."
            )
        step += 1

        chain.append(
            f"Step {step} [Confidence]   {result.confidence:.2f} "
            f"({len(result.matched_rules)} rule(s) fired, "
            f"{len(primary_rules)} supporting primary category). "
            + self._confidence_explanation(result)
        )
        step += 1

        ctx = result.task_context
        severity_source = "rule signals"
        if ctx:
            if ctx.task_completed is False and result.severity == IssueSeverity.HIGH:
                chain.append(
                    f"Step {step} [Context]      task_completed=False → severity escalated to HIGH. "
                    f"Task failure is the strongest severity signal available."
                )
                severity_source = "task failure context"
                step += 1
            elif (
                ctx.task_time_seconds
                and ctx.expected_time_seconds
                and ctx.task_time_seconds > ctx.expected_time_seconds * 1.5
            ):
                chain.append(
                    f"Step {step} [Context]      Task time {ctx.task_time_seconds:.0f}s exceeded "
                    f"expected {ctx.expected_time_seconds:.0f}s — severity escalated."
                )
                step += 1

        chain.append(
            f"Step {step} [Severity]     Final: {result.severity.value} "
            f"(determined by {severity_source})."
        )
        step += 1

        hm = self._map_to_heuristic(result.issue_category)
        if hm and hm.primary_number:
            sec = f" + H{hm.secondary_number}: {hm.secondary_name}" if hm.secondary_number else ""
            chain.append(
                f"Step {step} [Heuristic]    '{result.issue_category.value}' → "
                f"Nielsen H{hm.primary_number}: {hm.primary_name}{sec}. "
                f"Mapping confidence: {hm.confidence}."
            )
        elif result.issue_category == IssueCategory.POSITIVE_SIGNAL:
            chain.append(f"Step {step} [Heuristic]    Positive signal — no heuristic violated.")
        else:
            chain.append(f"Step {step} [Heuristic]    Heuristic mapping unavailable.")
        step += 1

        rec = _get_recommendation(result.issue_category, result.severity)
        chain.append(
            f"Step {step} [Recommendation] "
            f"{result.issue_category.value.upper()} / {result.severity.value}: {rec}"
        )
        return chain

    def _map_to_heuristic(self, category: IssueCategory) -> HeuristicMatch:
        keyword = _CATEGORY_TO_KEYWORD.get(category, category.value)
        return NielsenMapper().map(keyword)

    def _format_ref(self, hm: Optional[HeuristicMatch], *, primary: bool) -> Optional[str]:
        if hm is None:
            return None
        num = hm.primary_number if primary else hm.secondary_number
        name = hm.primary_name if primary else hm.secondary_name
        if num is None:
            return None
        return f"Nielsen Heuristic {num}: {name}"

    def _format_desc(self, hm: Optional[HeuristicMatch]) -> Optional[str]:
        if hm is None:
            return None
        return hm.primary_description

    def _confidence_explanation(self, result: AnalysisResult) -> str:
        c, n = result.confidence, len(result.matched_rules)
        if c >= 0.80:
            return f"High confidence: {n} rule(s) with strong category agreement."
        if c >= 0.50:
            return f"Medium confidence: {n} rule(s) fired; some cross-category signals present."
        if c >= 0.25:
            return f"Low confidence: weak signal ({n} rule(s)); recommend manual review."
        return "Very low confidence: no strong pattern matched; result is a heuristic fallback."


# ══════════════════════════════════════════════════════════════════════════════
# STUDY-LEVEL AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StudyExplainabilityReport:
    """
    Aggregate explainability report for an entire RUXAILAB study.
    Designed to feed FinalReportView analytics and serialise to Firestore.
    """

    total_analyzed: int
    usability_issue_count: int
    high_priority_count: int
    category_distribution: dict[str, int]
    severity_distribution: dict[str, int]
    heuristic_frequency: dict[int, int]
    top_issues: list[ExplainedResult]
    recommendations_deduplicated: list[str]
    overall_sentiment_ratio: dict[str, int]
    all_results: list[ExplainedResult]

    def as_dict(self, include_all_results: bool = False) -> dict:
        d = {
            "total_analyzed": self.total_analyzed,
            "usability_issue_count": self.usability_issue_count,
            "high_priority_count": self.high_priority_count,
            "category_distribution": self.category_distribution,
            "severity_distribution": self.severity_distribution,
            "heuristic_frequency": {str(k): v for k, v in self.heuristic_frequency.items()},
            "top_issues": [r.as_dict() for r in self.top_issues],
            "recommendations": self.recommendations_deduplicated,
            "sentiment": self.overall_sentiment_ratio,
        }
        if include_all_results:
            d["all_results"] = [r.as_dict() for r in self.all_results]
        return d


class StudyExplainabilityAggregator:
    """
    Aggregates a list of ExplainedResults into a StudyExplainabilityReport.
    """

    def aggregate(self, explained: list[ExplainedResult]) -> StudyExplainabilityReport:
        if not explained:
            return StudyExplainabilityReport(
                total_analyzed=0,
                usability_issue_count=0,
                high_priority_count=0,
                category_distribution={},
                severity_distribution={},
                heuristic_frequency={},
                top_issues=[],
                recommendations_deduplicated=[],
                overall_sentiment_ratio={"negative": 0, "neutral": 0, "positive": 0},
                all_results=[],
            )

        issues = [e for e in explained if e.analysis.is_usability_issue()]

        cat_dist: dict[str, int] = {}
        for e in issues:
            k = e.analysis.issue_category.value
            cat_dist[k] = cat_dist.get(k, 0) + 1

        sev_dist: dict[str, int] = {}
        for e in issues:
            k = e.analysis.severity.value
            sev_dist[k] = sev_dist.get(k, 0) + 1

        h_freq: dict[int, int] = {}
        for e in issues:
            if e.nielsen_ref:
                num = _extract_heuristic_number(e.nielsen_ref)
                if num:
                    h_freq[num] = h_freq.get(num, 0) + 1

        _sev_rank = {
            IssueSeverity.HIGH: 3,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 1,
            IssueSeverity.NONE: 0,
        }
        top = sorted(
            issues,
            key=lambda e: (_sev_rank.get(e.analysis.severity, 0), e.analysis.confidence),
            reverse=True,
        )[:5]

        rec_counts: dict[str, int] = {}
        for e in issues:
            rec_counts[e.recommendation] = rec_counts.get(e.recommendation, 0) + 1
        recs = [r for r, _ in sorted(rec_counts.items(), key=lambda x: -x[1])]

        sentiment: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
        for e in explained:
            sentiment[e.analysis.sentiment.value] += 1

        return StudyExplainabilityReport(
            total_analyzed=len(explained),
            usability_issue_count=len(issues),
            high_priority_count=sum(1 for e in explained if e.is_high_priority),
            category_distribution=cat_dist,
            severity_distribution=sev_dist,
            heuristic_frequency=dict(sorted(h_freq.items())),
            top_issues=top,
            recommendations_deduplicated=recs,
            overall_sentiment_ratio=sentiment,
            all_results=explained,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


def analyze_and_explain(
    text: str,
    context: Optional[TaskContext] = None,
) -> ExplainedResult:
    """One-shot: analyze → explain."""
    result = SemanticAnalyzer().analyze(text, context)
    return ExplainabilityEngine().explain(result)


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _is_negated(match_start: int, text: str, negation_positions: set[int]) -> bool:
    if not negation_positions:
        return False
    preceding = text[:match_start]
    words_before = preceding.split()
    if not words_before:
        return False
    window_start = max(0, match_start - sum(len(w) + 1 for w in words_before[-_NEGATION_WINDOW:]))
    return any(window_start <= pos < match_start for pos in negation_positions)


def _determine_categories(
    fired_rules: list[MatchedRule],
) -> tuple[IssueCategory, Optional[IssueCategory]]:
    if not fired_rules:
        return IssueCategory.UNKNOWN, None
    scores: dict[IssueCategory, float] = {}
    for rule in fired_rules:
        scores[rule.category] = scores.get(rule.category, 0.0) + rule.confidence_contribution
    sorted_cats = sorted(scores.items(), key=lambda x: -x[1])
    primary = sorted_cats[0][0]
    secondary = sorted_cats[1][0] if len(sorted_cats) > 1 and sorted_cats[1][1] >= 0.2 else None
    if secondary == primary:
        secondary = sorted_cats[2][0] if len(sorted_cats) > 2 else None
    return primary, secondary


def _compute_confidence(
    fired_rules: list[MatchedRule],
    primary_category: IssueCategory,
) -> float:
    if not fired_rules:
        return 0.0
    primary_score = sum(r.confidence_contribution for r in fired_rules if r.category == primary_category)
    total_score = sum(r.confidence_contribution for r in fired_rules)
    competing = total_score - primary_score
    penalty = min(0.20, competing * 0.15)
    return round(max(0.0, min(1.0, primary_score) - penalty), 3)


def _determine_severity(
    fired_rules: list[MatchedRule],
    category: IssueCategory,
    context: Optional[TaskContext],
    confidence: float,
) -> IssueSeverity:
    if category in (IssueCategory.POSITIVE_SIGNAL, IssueCategory.UNKNOWN):
        return IssueSeverity.NONE
    if context and context.task_completed is False:
        return IssueSeverity.HIGH
    signal_order = [IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]
    for level in signal_order:
        if any(r.severity_signal == level for r in fired_rules):
            if (
                context
                and context.task_time_seconds
                and context.expected_time_seconds
                and context.task_time_seconds > context.expected_time_seconds * 1.5
                and level == IssueSeverity.LOW
            ):
                return IssueSeverity.MEDIUM
            return level
    if confidence >= 0.60:
        return IssueSeverity.MEDIUM
    if confidence >= 0.25:
        return IssueSeverity.LOW
    return IssueSeverity.NONE


def _determine_sentiment(
    fired_rules: list[MatchedRule],
    category: IssueCategory,
) -> IssueSentiment:
    if category == IssueCategory.POSITIVE_SIGNAL:
        return IssueSentiment.POSITIVE
    neg = sum(1 for r in fired_rules if r.sentiment_signal == IssueSentiment.NEGATIVE)
    pos = sum(1 for r in fired_rules if r.sentiment_signal == IssueSentiment.POSITIVE)
    if neg > pos:
        return IssueSentiment.NEGATIVE
    if pos > neg:
        return IssueSentiment.POSITIVE
    if category != IssueCategory.UNKNOWN:
        return IssueSentiment.NEGATIVE
    return IssueSentiment.NEUTRAL


def _extract_heuristic_number(nielsen_ref: str) -> Optional[int]:
    m = re.search(r"Heuristic\s+(\d+)", nielsen_ref)
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (PYTHONPATH=. python explainability.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEP = "─" * 70

    analyzer = SemanticAnalyzer()
    engine = ExplainabilityEngine()
    agg = StudyExplainabilityAggregator()

    test_cases = [
        (
            "i could not find the submit button anywhere on the page",
            TaskContext(task_id="t1", task_completed=False),
        ),
        (
            "the loading took forever and nothing happened after i clicked save",
            TaskContext(task_id="t2", task_completed=False, task_time_seconds=120, expected_time_seconds=30),
        ),
        (
            "i did not understand the error message it said something about a 503",
            TaskContext(task_id="t3", task_completed=False),
        ),
        (
            "there are too many fields on this form it is overwhelming",
            TaskContext(task_id="t4", task_completed=True),
        ),
        (
            "i expected the back button to undo my last action but it navigated away",
            TaskContext(task_id="t5", task_completed=True),
        ),
        (
            "the page was really slow and kept freezing when i tried to upload",
            TaskContext(task_id="t6", task_completed=False),
        ),
        (
            "i do not trust this site it looks unofficial and asks for my card details",
            TaskContext(task_id="t7", task_completed=False),
        ),
        (
            "everything was intuitive and i found what i needed right away",
            TaskContext(task_id="t8", task_completed=True),
        ),
    ]

    all_explained: list[ExplainedResult] = []

    for text, ctx in test_cases:
        result = analyzer.analyze(text, ctx)
        explained = engine.explain(result)
        all_explained.append(explained)

        print(SEP)
        print(f"INPUT    : {text}")
        print(
            f"CATEGORY : {result.issue_category.value:<20}  "
            f"SEVERITY: {result.severity.value:<8}  "
            f"CONF: {result.confidence:.2f}  "
            f"SENTIMENT: {result.sentiment.value}"
        )
        print(f"HEURISTIC: {explained.nielsen_ref or '(none)'}")
        print("EVIDENCE:")
        for e in explained.evidence[:3]:
            print(f"  • {e}")
        print("REASONING:")
        for step in explained.reasoning_chain:
            print(f"  {step}")
        print(f"PRIORITY : {'HIGH PRIORITY' if explained.is_high_priority else 'normal'}")

    print("\n" + "=" * 70)
    print("STUDY AGGREGATE")
    print("=" * 70)
    report = agg.aggregate(all_explained)
    print(f"Total          : {report.total_analyzed}")
    print(f"Issues         : {report.usability_issue_count}")
    print(f"High priority  : {report.high_priority_count}")
    print(f"Categories     : {report.category_distribution}")
    print(f"Severity       : {report.severity_distribution}")
    print(f"Heuristics     : {report.heuristic_frequency}")
    print(f"Sentiment      : {report.overall_sentiment_ratio}")
    if report.recommendations_deduplicated:
        print(f"Top rec        : {report.recommendations_deduplicated[0][:100]}…")
