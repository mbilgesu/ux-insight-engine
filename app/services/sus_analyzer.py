from __future__ import annotations

from app.services.ruxailab_methodology import calculate_sus_score, sus_adjective_for_score


def sus_score(responses: list[int]) -> float:
    return calculate_sus_score(responses).score


def sus_adjective_rating(score: float) -> str:
    return sus_adjective_for_score(score)
