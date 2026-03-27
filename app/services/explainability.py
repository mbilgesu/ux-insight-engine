from __future__ import annotations

from dataclasses import dataclass

from app.services.nielsen_mapper import NielsenMapping
from app.services.semantic_analysis import AnalysisResult


@dataclass(frozen=True)
class ExplainedResult:
    record_id: str
    summary: str
    reasoning: str
    recommendation: str


_RECOMMENDATIONS: dict[str, str] = {
    "navigation": (
        "Surface key actions in primary navigation or persistent chrome; "
        "add wayfinding labels and predictable entry points."
    ),
    "terminology": (
        "Align labels and microcopy with user vocabulary; test terms in quick comprehension checks."
    ),
    "visual_hierarchy": (
        "Increase visual prominence of primary actions; improve contrast, spacing, and scanning paths."
    ),
    "feedback_missing": (
        "Add explicit system status after actions (loading, success, failure); keep feedback adjacent to the control."
    ),
    "form_friction": (
        "Reduce required fields, preserve input on error, and clarify validation inline with the field."
    ),
    "performance": (
        "Address perceived latency (skeletons, optimistic UI) and measure slow paths for real optimization."
    ),
    "trust_clarity": (
        "Clarify security, pricing, and data use; add recognizable trust signals where users hesitate."
    ),
    "error_recovery": (
        "Offer recovery paths, plain-language errors, and undo where destructive actions exist."
    ),
}


def explain(
    analysis: AnalysisResult,
    nielsen: NielsenMapping | None = None,
) -> ExplainedResult:
    cat = analysis.issue_category or "general usability"
    sev = analysis.severity
    sent = analysis.sentiment_label
    task = analysis.task_signal

    nielsen_bits = ""
    if nielsen and nielsen.heuristics:
        parts = [f"H{h.number}: {h.title}" for h in nielsen.heuristics]
        nielsen_bits = " Mapped heuristics: " + "; ".join(parts) + "."

    reasoning = (
        f"Detected signals point to {cat} with {sent} tone and {task} task pattern; "
        f"severity {sev} given overlap of evidence.{nielsen_bits} "
        f"Rules fired: {len(analysis.matched_rules)} (confidence {analysis.confidence:.2f})."
    )

    summary = f"[{sev.upper()}] {cat.replace('_', ' ')} — {task.replace('_', ' ')} ({sent})"

    rec = _RECOMMENDATIONS.get(analysis.issue_category or "", "") or (
        "Review the utterance with a short usability pass and validate with one targeted user task."
    )

    return ExplainedResult(
        record_id=analysis.record_id,
        summary=summary.strip(),
        reasoning=reasoning.strip(),
        recommendation=rec,
    )
