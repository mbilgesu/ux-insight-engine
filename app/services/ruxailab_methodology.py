"""
Modules 2–4 of the RUXAILAB NLP stack: Nielsen mapping, SUS, NASA-TLX.
Loaded by ruxailab_nlp.py for a single import surface.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "NielsenHeuristic",
    "HeuristicMatch",
    "NielsenMapper",
    "map_to_heuristics",
    "SUSResult",
    "AggregatedSUSResult",
    "SUSAnalyzer",
    "calculate_sus_score",
    "aggregate_sus_scores",
    "sus_adjective_for_score",
    "TLXDimension",
    "TLXWeights",
    "NASATLXResult",
    "AggregatedNASATLXResult",
    "NASATLXAnalyzer",
    "analyze_nasa_tlx",
    "aggregate_nasa_tlx",
    "TLX_DIMENSION_KEYS",
    "CATEGORY_DIRECT_MAP",
]


class NielsenHeuristic(IntEnum):
    VISIBILITY_OF_SYSTEM_STATUS = 1
    MATCH_REAL_WORLD = 2
    USER_CONTROL_AND_FREEDOM = 3
    CONSISTENCY_AND_STANDARDS = 4
    ERROR_PREVENTION = 5
    RECOGNITION_OVER_RECALL = 6
    FLEXIBILITY_AND_EFFICIENCY = 7
    AESTHETIC_AND_MINIMALIST_DESIGN = 8
    HELP_WITH_ERRORS = 9
    HELP_AND_DOCUMENTATION = 10


_HEURISTIC_META: dict[NielsenHeuristic, dict] = {
    NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS: {
        "name": "Visibility of System Status",
        "description": "Keep users informed about what is going on through appropriate feedback within reasonable time.",
        "examples": ["loading indicators", "progress bars", "status messages", "upload feedback"],
    },
    NielsenHeuristic.MATCH_REAL_WORLD: {
        "name": "Match Between System and the Real World",
        "description": "Speak the users' language; use familiar words, phrases, and real-world conventions.",
        "examples": ["familiar terminology", "real-world metaphors", "plain language"],
    },
    NielsenHeuristic.USER_CONTROL_AND_FREEDOM: {
        "name": "User Control and Freedom",
        "description": "Provide clearly marked emergency exits, undo, and redo for mistaken actions.",
        "examples": ["undo/redo", "cancel buttons", "back navigation", "easy exit"],
    },
    NielsenHeuristic.CONSISTENCY_AND_STANDARDS: {
        "name": "Consistency and Standards",
        "description": "Users should not have to wonder whether different words or actions mean the same thing.",
        "examples": ["consistent labeling", "platform conventions", "predictable interactions"],
    },
    NielsenHeuristic.ERROR_PREVENTION: {
        "name": "Error Prevention",
        "description": "Design that prevents problems from occurring — better than good error messages.",
        "examples": ["input validation", "confirmation dialogs", "constraints on input"],
    },
    NielsenHeuristic.RECOGNITION_OVER_RECALL: {
        "name": "Recognition Rather Than Recall",
        "description": "Minimize memory load by making objects, actions, and options visible.",
        "examples": ["visible options", "menus over commands", "breadcrumbs", "recent items"],
    },
    NielsenHeuristic.FLEXIBILITY_AND_EFFICIENCY: {
        "name": "Flexibility and Efficiency of Use",
        "description": "Accelerators speed up interaction for experts while remaining invisible to novices.",
        "examples": ["keyboard shortcuts", "customization", "advanced search"],
    },
    NielsenHeuristic.AESTHETIC_AND_MINIMALIST_DESIGN: {
        "name": "Aesthetic and Minimalist Design",
        "description": "Dialogues should not contain irrelevant information that competes with relevant content.",
        "examples": ["clean layout", "reduced clutter", "focused content"],
    },
    NielsenHeuristic.HELP_WITH_ERRORS: {
        "name": "Help Users Recognize, Diagnose, and Recover from Errors",
        "description": "Error messages in plain language, indicating the problem and suggesting a solution.",
        "examples": ["plain error messages", "error recovery steps", "actionable suggestions"],
    },
    NielsenHeuristic.HELP_AND_DOCUMENTATION: {
        "name": "Help and Documentation",
        "description": "Documentation should be easy to search and focused on the user's task.",
        "examples": ["searchable help", "tooltips", "onboarding", "task-focused docs"],
    },
}

CATEGORY_DIRECT_MAP: dict[str, tuple[NielsenHeuristic, Optional[NielsenHeuristic]]] = {
    "navigation": (NielsenHeuristic.RECOGNITION_OVER_RECALL, NielsenHeuristic.USER_CONTROL_AND_FREEDOM),
    "terminology": (NielsenHeuristic.MATCH_REAL_WORLD, None),
    "visual_hierarchy": (
        NielsenHeuristic.AESTHETIC_AND_MINIMALIST_DESIGN,
        NielsenHeuristic.RECOGNITION_OVER_RECALL,
    ),
    "feedback_missing": (NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS, None),
    "form_friction": (NielsenHeuristic.AESTHETIC_AND_MINIMALIST_DESIGN, NielsenHeuristic.ERROR_PREVENTION),
    "performance": (NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS, None),
    "trust_clarity": (NielsenHeuristic.HELP_AND_DOCUMENTATION, NielsenHeuristic.ERROR_PREVENTION),
    "error_recovery": (NielsenHeuristic.HELP_WITH_ERRORS, NielsenHeuristic.USER_CONTROL_AND_FREEDOM),
}

_NM_RULES: list[tuple[str, NielsenHeuristic, Optional[NielsenHeuristic]]] = [
    (r"\bload(ing)?\b", NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS, None),
    (r"\bprogress\b", NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS, None),
    (r"\bfeedback\b", NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS, None),
    (r"\bjargon\b", NielsenHeuristic.MATCH_REAL_WORLD, None),
    (r"\bterminolog(y|ies)\b", NielsenHeuristic.MATCH_REAL_WORLD, None),
    (r"\bundo\b", NielsenHeuristic.USER_CONTROL_AND_FREEDOM, None),
    (r"\bcancel\b", NielsenHeuristic.USER_CONTROL_AND_FREEDOM, None),
    (r"\bnavigation\b", NielsenHeuristic.USER_CONTROL_AND_FREEDOM, NielsenHeuristic.RECOGNITION_OVER_RECALL),
    (r"\bstuck\b", NielsenHeuristic.USER_CONTROL_AND_FREEDOM, None),
    (r"\bconsisten(cy|t)\b", NielsenHeuristic.CONSISTENCY_AND_STANDARDS, None),
    (r"\bvalidat(e|ion)\b", NielsenHeuristic.ERROR_PREVENTION, None),
    (r"\bsearch\b", NielsenHeuristic.RECOGNITION_OVER_RECALL, None),
    (r"\bmenu(s)?\b", NielsenHeuristic.RECOGNITION_OVER_RECALL, None),
    (r"\bshortcut(s)?\b", NielsenHeuristic.FLEXIBILITY_AND_EFFICIENCY, None),
    (r"\bclutter\b", NielsenHeuristic.AESTHETIC_AND_MINIMALIST_DESIGN, None),
    (r"\berror\s+message\b", NielsenHeuristic.HELP_WITH_ERRORS, None),
    (r"\bcrash\b", NielsenHeuristic.HELP_WITH_ERRORS, NielsenHeuristic.VISIBILITY_OF_SYSTEM_STATUS),
    (r"\bdocumentation\b", NielsenHeuristic.HELP_AND_DOCUMENTATION, None),
    (r"\bonboard(ing)?\b", NielsenHeuristic.HELP_AND_DOCUMENTATION, None),
]

_NM_COMPILED = [(re.compile(p, re.IGNORECASE), prim, sec) for p, prim, sec in _NM_RULES]

_NM_FALLBACKS: list[tuple[str, NielsenHeuristic, Optional[NielsenHeuristic]]] = [
    (r"\berror\b", NielsenHeuristic.HELP_WITH_ERRORS, NielsenHeuristic.ERROR_PREVENTION),
    (r"\bhelp\b", NielsenHeuristic.HELP_AND_DOCUMENTATION, None),
    (r"\bconfus(ed|ing)?\b", NielsenHeuristic.MATCH_REAL_WORLD, NielsenHeuristic.RECOGNITION_OVER_RECALL),
]


@dataclass
class HeuristicMatch:
    input_category: str
    primary: Optional[NielsenHeuristic]
    secondary: Optional[NielsenHeuristic] = None
    confidence: Optional[str] = None
    matched_terms: list[str] = field(default_factory=list)
    is_fallback: bool = False

    @property
    def primary_number(self) -> Optional[int]:
        return int(self.primary) if self.primary else None

    @property
    def primary_name(self) -> Optional[str]:
        return _HEURISTIC_META[self.primary]["name"] if self.primary else None

    @property
    def secondary_number(self) -> Optional[int]:
        return int(self.secondary) if self.secondary else None

    @property
    def secondary_name(self) -> Optional[str]:
        return _HEURISTIC_META[self.secondary]["name"] if self.secondary else None

    def as_dict(self) -> dict:
        return {
            "input_category": self.input_category,
            "primary_heuristic": self.primary_number,
            "primary_name": self.primary_name,
            "secondary_heuristic": self.secondary_number,
            "secondary_name": self.secondary_name,
            "confidence": self.confidence,
            "matched_terms": self.matched_terms,
            "is_fallback": self.is_fallback,
        }


class NielsenMapper:
    def map(self, category: str, *, enable_fallback: bool = True) -> HeuristicMatch:
        if category in CATEGORY_DIRECT_MAP:
            prim, sec = CATEGORY_DIRECT_MAP[category]
            return HeuristicMatch(
                input_category=category,
                primary=prim,
                secondary=sec,
                confidence="high",
                matched_terms=[category],
                is_fallback=False,
            )

        norm = re.sub(r"[_\-/]+", " ", (category or "").strip().lower())
        norm = re.sub(r"\s+", " ", norm).strip()

        primary: Optional[NielsenHeuristic] = None
        secondary: Optional[NielsenHeuristic] = None
        matched: list[str] = []
        confidence: Optional[str] = None

        for pattern, rule_prim, rule_sec in _NM_COMPILED:
            m = pattern.search(norm)
            if not m:
                continue
            if primary is None:
                primary, secondary = rule_prim, rule_sec
                matched.append(m.group(0))
                confidence = "high"
            elif primary == rule_prim and secondary is None and rule_sec:
                secondary = rule_sec
                matched.append(m.group(0))
            elif rule_prim not in (primary, secondary) and secondary is None:
                secondary = rule_prim
                matched.append(m.group(0))
                confidence = "medium"

        if primary is not None:
            return HeuristicMatch(category, primary, secondary, confidence, matched, False)

        if enable_fallback:
            for p_str, prim, sec in _NM_FALLBACKS:
                m = re.search(p_str, norm, re.IGNORECASE)
                if m:
                    return HeuristicMatch(category, prim, sec, "low", [m.group(0)], True)

        return HeuristicMatch(category, None, None, None, [], False)

    def map_batch(self, categories: list[str], *, enable_fallback: bool = True) -> list[HeuristicMatch]:
        return [self.map(c, enable_fallback=enable_fallback) for c in categories]

    @staticmethod
    def all_heuristics() -> list[dict]:
        return [{"number": int(h), **_HEURISTIC_META[h]} for h in NielsenHeuristic]


def map_to_heuristics(category: str, *, enable_fallback: bool = True) -> HeuristicMatch:
    return NielsenMapper().map(category, enable_fallback=enable_fallback)


_SUS_PASSING = 68.0
_SUS_ITEMS = 10


@dataclass
class SUSResult:
    raw_responses: list[int]
    score: float
    adjective_rating: str
    acceptability: str
    letter_grade: str
    passes_threshold: bool
    percentile: int
    odd_item_scores: list[int]
    even_item_scores: list[int]
    warnings: list[str] = field(default_factory=list)

    @property
    def interpretation(self) -> str:
        return (
            f"SUS score {self.score:.1f} ({self.adjective_rating}) — "
            f"Grade {self.letter_grade} — {self.acceptability} — "
            f"~{self.percentile}th percentile."
        )

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "adjective_rating": self.adjective_rating,
            "acceptability": self.acceptability,
            "letter_grade": self.letter_grade,
            "passes_threshold": self.passes_threshold,
            "percentile": self.percentile,
            "interpretation": self.interpretation,
            "warnings": self.warnings,
        }


@dataclass
class AggregatedSUSResult:
    scores: list[float]
    mean_score: float
    median_score: float
    std_dev: float
    min_score: float
    max_score: float
    adjective_rating: str
    acceptability: str
    letter_grade: str
    passes_threshold: bool
    percentile: int
    respondent_count: int
    cronbach_alpha: Optional[float]
    score_distribution: dict[str, int]
    individual_results: list[SUSResult]

    @property
    def interpretation(self) -> str:
        alpha_str = f" Cronbach α={self.cronbach_alpha:.2f}." if self.cronbach_alpha is not None else ""
        return (
            f"Mean SUS {self.mean_score:.1f} across {self.respondent_count} respondent(s) "
            f"({self.adjective_rating}, Grade {self.letter_grade}, {self.acceptability}). "
            f"SD={self.std_dev:.1f}.{alpha_str}"
        )

    def as_dict(self) -> dict:
        return {
            "mean_score": round(self.mean_score, 2),
            "median_score": round(self.median_score, 2),
            "std_dev": round(self.std_dev, 2),
            "min_score": round(self.min_score, 2),
            "max_score": round(self.max_score, 2),
            "adjective_rating": self.adjective_rating,
            "acceptability": self.acceptability,
            "letter_grade": self.letter_grade,
            "passes_threshold": self.passes_threshold,
            "percentile": self.percentile,
            "respondent_count": self.respondent_count,
            "cronbach_alpha": round(self.cronbach_alpha, 3) if self.cronbach_alpha is not None else None,
            "score_distribution": self.score_distribution,
            "interpretation": self.interpretation,
        }


class SUSAnalyzer:
    def analyze(self, responses: list) -> SUSResult:
        validated, warnings = _sus_validate(responses)
        odd = [validated[i] - 1 for i in range(0, 10, 2)]
        even = [5 - validated[i] for i in range(1, 10, 2)]
        score = (sum(odd) + sum(even)) * 2.5
        return SUSResult(
            raw_responses=list(responses),
            score=score,
            adjective_rating=_sus_adjective(score),
            acceptability=_sus_acceptability(score),
            letter_grade=_sus_grade(score),
            passes_threshold=score >= _SUS_PASSING,
            percentile=_sus_percentile(score),
            odd_item_scores=odd,
            even_item_scores=even,
            warnings=warnings,
        )

    def aggregate(self, all_responses: list[list]) -> AggregatedSUSResult:
        if not all_responses:
            raise ValueError("all_responses must not be empty.")
        individual = [self.analyze(r) for r in all_responses]
        scores = [r.score for r in individual]
        mean = _stat_mean(scores)
        return AggregatedSUSResult(
            scores=scores,
            mean_score=mean,
            median_score=_stat_median(scores),
            std_dev=_stat_stdev(scores),
            min_score=min(scores),
            max_score=max(scores),
            adjective_rating=_sus_adjective(mean),
            acceptability=_sus_acceptability(mean),
            letter_grade=_sus_grade(mean),
            passes_threshold=mean >= _SUS_PASSING,
            percentile=_sus_percentile(mean),
            respondent_count=len(individual),
            cronbach_alpha=_sus_cronbach(all_responses) if len(all_responses) >= 2 else None,
            score_distribution=_sus_distribution(scores),
            individual_results=individual,
        )


def calculate_sus_score(responses: list) -> SUSResult:
    return SUSAnalyzer().analyze(responses)


def aggregate_sus_scores(all_responses: list[list]) -> AggregatedSUSResult:
    return SUSAnalyzer().aggregate(all_responses)


def _sus_validate(responses: list) -> tuple[list[int], list[str]]:
    if len(responses) != _SUS_ITEMS:
        raise ValueError(f"SUS requires exactly {_SUS_ITEMS} responses, got {len(responses)}.")
    validated, warnings = [], []
    for i, val in enumerate(responses):
        try:
            v = int(float(val))
        except (TypeError, ValueError):
            warnings.append(f"Item {i + 1}: non-numeric {val!r} defaulted to 3.")
            v = 3
        if v < 1 or v > 5:
            warnings.append(f"Item {i + 1}: value {v} out of [1,5], clamped.")
            v = max(1, min(5, v))
        validated.append(v)
    return validated, warnings


def _sus_adjective(score: float) -> str:
    if score >= 85.5:
        return "Best Imaginable"
    if score >= 80.3:
        return "Excellent"
    if score >= 68.0:
        return "Good"
    if score >= 51.0:
        return "OK"
    if score >= 25.0:
        return "Poor"
    return "Worst Imaginable"


def sus_adjective_for_score(score: float) -> str:
    return _sus_adjective(score)


def _sus_acceptability(score: float) -> str:
    if score >= 70:
        return "Acceptable"
    if score >= 50:
        return "Marginal"
    return "Not Acceptable"


def _sus_grade(score: float) -> str:
    if score >= 90.9:
        return "A+"
    if score >= 85.5:
        return "A"
    if score >= 80.8:
        return "A-"
    if score >= 78.9:
        return "B+"
    if score >= 77.2:
        return "B"
    if score >= 74.1:
        return "B-"
    if score >= 72.6:
        return "C+"
    if score >= 71.1:
        return "C"
    if score >= 65.0:
        return "C-"
    if score >= 62.7:
        return "D+"
    if score >= 51.7:
        return "D"
    if score >= 51.0:
        return "D-"
    return "F"


def _sus_percentile(score: float) -> int:
    z = (score - 68.0) / 12.5
    pct = (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0 * 100
    return max(1, min(99, round(pct)))


def _sus_cronbach(all_responses: list[list]) -> Optional[float]:
    try:
        coerced = [_sus_validate(r)[0] for r in all_responses]
        k, n = _SUS_ITEMS, len(coerced)
        if n < 2:
            return None
        item_vars = [_stat_var_pop([coerced[r][i] for r in range(n)]) for i in range(k)]
        total_scores = [sum(r) for r in coerced]
        total_var = _stat_var_pop(total_scores)
        if total_var == 0:
            return None
        alpha = (k / (k - 1)) * (1.0 - sum(item_vars) / total_var)
        return round(max(0.0, min(1.0, alpha)), 4)
    except Exception:
        return None


def _sus_distribution(scores: list[float]) -> dict[str, int]:
    b = {
        "Best Imaginable (≥85.5)": 0,
        "Excellent (80.3–85.4)": 0,
        "Good (68.0–80.2)": 0,
        "OK (51.0–67.9)": 0,
        "Poor (25.0–50.9)": 0,
        "Worst Imaginable (<25)": 0,
    }
    for s in scores:
        if s >= 85.5:
            b["Best Imaginable (≥85.5)"] += 1
        elif s >= 80.3:
            b["Excellent (80.3–85.4)"] += 1
        elif s >= 68.0:
            b["Good (68.0–80.2)"] += 1
        elif s >= 51.0:
            b["OK (51.0–67.9)"] += 1
        elif s >= 25.0:
            b["Poor (25.0–50.9)"] += 1
        else:
            b["Worst Imaginable (<25)"] += 1
    return b


_TLX_KEYS = (
    "mentalDemand",
    "physicalDemand",
    "temporalDemand",
    "performance",
    "effort",
    "frustration",
)

TLX_DIMENSION_KEYS: Tuple[str, ...] = _TLX_KEYS

_TLX_LABELS = {
    "mentalDemand": "Mental Demand",
    "physicalDemand": "Physical Demand",
    "temporalDemand": "Temporal Demand",
    "performance": "Performance",
    "effort": "Effort",
    "frustration": "Frustration",
}

_TLX_WORKLOAD_THRESHOLDS = [(80, "Very High"), (60, "High"), (40, "Moderate"), (20, "Low"), (0, "Very Low")]


class TLXDimension:
    MENTAL_DEMAND = "mentalDemand"
    PHYSICAL_DEMAND = "physicalDemand"
    TEMPORAL_DEMAND = "temporalDemand"
    PERFORMANCE = "performance"
    EFFORT = "effort"
    FRUSTRATION = "frustration"


@dataclass
class TLXWeights:
    wins: dict[str, int]

    @classmethod
    def from_wins(cls, wins: dict[str, int]) -> TLXWeights:
        for k in _TLX_KEYS:
            if k not in wins:
                raise ValueError(f"TLXWeights.from_wins: missing dimension '{k}'.")
        if sum(wins.values()) != 15:
            raise ValueError(f"Win counts must sum to 15, got {sum(wins.values())}.")
        return cls(wins=dict(wins))

    @classmethod
    def uniform(cls) -> TLXWeights:
        return cls(wins={k: 1 for k in _TLX_KEYS})

    def normalized(self) -> dict[str, float]:
        total = sum(self.wins.values())
        if total == 0:
            return {k: 1.0 / len(_TLX_KEYS) for k in _TLX_KEYS}
        return {k: self.wins[k] / total for k in _TLX_KEYS}

    @property
    def is_uniform(self) -> bool:
        return len(set(self.wins.values())) == 1


@dataclass
class _DimScore:
    key: str
    label: str
    raw_score: float
    weighted_contribution: float
    workload_level: str


@dataclass
class NASATLXResult:
    raw_responses: dict[str, float]
    overall_score: float
    rtlx_score: float
    weights_used: TLXWeights
    is_weighted: bool
    dominant_dimension: str
    least_demanding_dimension: str
    workload_level: str
    dimension_scores: list[_DimScore]
    warnings: list[str] = field(default_factory=list)

    @property
    def interpretation(self) -> str:
        mode = "Weighted TLX" if self.is_weighted else "Raw TLX (rTLX)"
        return (
            f"{mode} workload: {self.overall_score:.1f}/100 ({self.workload_level}). "
            f"Dominant: {_TLX_LABELS[self.dominant_dimension]}. "
            f"Least demanding: {_TLX_LABELS[self.least_demanding_dimension]}."
        )

    def as_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 2),
            "rtlx_score": round(self.rtlx_score, 2),
            "is_weighted": self.is_weighted,
            "workload_level": self.workload_level,
            "dominant_dimension": self.dominant_dimension,
            "least_demanding_dimension": self.least_demanding_dimension,
            "dimensions": [
                {
                    "key": d.key,
                    "label": d.label,
                    "raw_score": round(d.raw_score, 2),
                    "weighted_contribution": round(d.weighted_contribution, 2),
                    "workload_level": d.workload_level,
                }
                for d in self.dimension_scores
            ],
            "interpretation": self.interpretation,
            "warnings": self.warnings,
        }


@dataclass
class AggregatedNASATLXResult:
    mean_overall_score: float
    std_dev: float
    min_score: float
    max_score: float
    rtlx_mean: float
    is_weighted: bool
    workload_level: str
    respondent_count: int
    dimension_averages: dict[str, float]
    dimension_std_devs: dict[str, float]
    most_stressful_dimension: str
    least_stressful_dimension: str
    high_workload_count: int
    burnout_risk: str
    individual_results: list[NASATLXResult]

    @property
    def interpretation(self) -> str:
        mode = "weighted" if self.is_weighted else "unweighted rTLX"
        return (
            f"Mean {mode} workload {self.mean_overall_score:.1f} ({self.workload_level}) "
            f"across {self.respondent_count} respondent(s). SD={self.std_dev:.1f}. "
            f"Most stressful: {_TLX_LABELS[self.most_stressful_dimension]}. "
            f"{self.high_workload_count} respondent(s) scored ≥ 60. "
            f"Burnout risk: {self.burnout_risk}."
        )

    def as_dict(self) -> dict:
        return {
            "mean_overall_score": round(self.mean_overall_score, 2),
            "std_dev": round(self.std_dev, 2),
            "min_score": round(self.min_score, 2),
            "max_score": round(self.max_score, 2),
            "rtlx_mean": round(self.rtlx_mean, 2),
            "is_weighted": self.is_weighted,
            "workload_level": self.workload_level,
            "respondent_count": self.respondent_count,
            "dimension_averages": {k: round(v, 2) for k, v in self.dimension_averages.items()},
            "dimension_std_devs": {k: round(v, 2) for k, v in self.dimension_std_devs.items()},
            "most_stressful_dimension": self.most_stressful_dimension,
            "least_stressful_dimension": self.least_stressful_dimension,
            "high_workload_count": self.high_workload_count,
            "burnout_risk": self.burnout_risk,
            "interpretation": self.interpretation,
        }


class NASATLXAnalyzer:
    def analyze(self, responses: dict[str, float], weights: Optional[TLXWeights] = None) -> NASATLXResult:
        validated, warnings = _tlx_validate(responses)
        weights_used = weights or TLXWeights.uniform()
        is_weighted = not weights_used.is_uniform
        norm_w = weights_used.normalized()

        rtlx = sum(validated[k] for k in _TLX_KEYS) / len(_TLX_KEYS)
        overall = sum(validated[k] * norm_w[k] for k in _TLX_KEYS)

        dims = [
            _DimScore(
                key=k,
                label=_TLX_LABELS[k],
                raw_score=validated[k],
                weighted_contribution=validated[k] * norm_w[k],
                workload_level=_tlx_workload_level(validated[k]),
            )
            for k in _TLX_KEYS
        ]
        dominant = max(dims, key=lambda d: d.weighted_contribution).key
        least = min(dims, key=lambda d: d.raw_score).key

        return NASATLXResult(
            raw_responses=dict(validated),
            overall_score=overall,
            rtlx_score=rtlx,
            weights_used=weights_used,
            is_weighted=is_weighted,
            dominant_dimension=dominant,
            least_demanding_dimension=least,
            workload_level=_tlx_workload_level(overall),
            dimension_scores=dims,
            warnings=warnings,
        )

    def aggregate(self, all_responses: list[dict], weights: Optional[TLXWeights] = None) -> AggregatedNASATLXResult:
        if not all_responses:
            raise ValueError("all_responses must not be empty.")
        individual = [self.analyze(r, weights) for r in all_responses]
        overall_scores = [r.overall_score for r in individual]
        dim_avgs = {k: _stat_mean([r.raw_responses[k] for r in individual]) for k in _TLX_KEYS}
        dim_sds = {k: _stat_stdev([r.raw_responses[k] for r in individual]) for k in _TLX_KEYS}
        mean = _stat_mean(overall_scores)

        return AggregatedNASATLXResult(
            mean_overall_score=mean,
            std_dev=_stat_stdev(overall_scores),
            min_score=min(overall_scores),
            max_score=max(overall_scores),
            rtlx_mean=_stat_mean([r.rtlx_score for r in individual]),
            is_weighted=any(r.is_weighted for r in individual),
            workload_level=_tlx_workload_level(mean),
            respondent_count=len(individual),
            dimension_averages=dim_avgs,
            dimension_std_devs=dim_sds,
            most_stressful_dimension=max(_TLX_KEYS, key=lambda k: dim_avgs[k]),
            least_stressful_dimension=min(_TLX_KEYS, key=lambda k: dim_avgs[k]),
            high_workload_count=sum(1 for s in overall_scores if s >= 60),
            burnout_risk=_tlx_burnout_risk(mean, _stat_stdev(overall_scores)),
            individual_results=individual,
        )


def analyze_nasa_tlx(responses: dict[str, float], weights: Optional[TLXWeights] = None) -> NASATLXResult:
    return NASATLXAnalyzer().analyze(responses, weights)


def aggregate_nasa_tlx(all_responses: list[dict], weights: Optional[TLXWeights] = None) -> AggregatedNASATLXResult:
    return NASATLXAnalyzer().aggregate(all_responses, weights)


def _tlx_validate(responses: dict) -> tuple[dict[str, float], list[str]]:
    out, warnings = {}, []
    for k in _TLX_KEYS:
        raw = responses.get(k)
        if raw is None:
            warnings.append(f"Dimension '{k}' missing; defaulted to 0.")
            out[k] = 0.0
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"Dimension '{k}': non-numeric {raw!r} defaulted to 0.")
            out[k] = 0.0
            continue
        if v < 0 or v > 100:
            warnings.append(f"Dimension '{k}': {v} out of [0,100], clamped.")
            v = max(0.0, min(100.0, v))
        out[k] = v
    return out, warnings


def _tlx_workload_level(score: float) -> str:
    for threshold, label in _TLX_WORKLOAD_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Low"


def _tlx_burnout_risk(mean: float, sd: float) -> str:
    if mean >= 80 or (mean >= 60 and sd >= 20):
        return "High"
    if mean >= 60 or (mean >= 50 and sd >= 15):
        return "Moderate"
    return "Low"


def _stat_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stat_median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else float(s[mid])


def _stat_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _stat_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _stat_var_pop(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _stat_mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)
