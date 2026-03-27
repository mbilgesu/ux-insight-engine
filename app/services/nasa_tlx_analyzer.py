from __future__ import annotations

from typing import Sequence

from app.services.ruxailab_methodology import (
    TLX_DIMENSION_KEYS,
    TLXWeights,
    analyze_nasa_tlx,
)


def nasa_tlx_unweighted(dimensions: Sequence[float]) -> float:
    if len(dimensions) != 6:
        raise ValueError("NASA-TLX requires exactly 6 dimension scores")
    d = {k: float(dimensions[i]) for i, k in enumerate(TLX_DIMENSION_KEYS)}
    return analyze_nasa_tlx(d, TLXWeights.uniform()).rtlx_score


def nasa_tlx_weighted(dimensions: Sequence[float], weights: Sequence[float]) -> float:
    if len(dimensions) != 6 or len(weights) != 6:
        raise ValueError("dimensions and weights must each have length 6")
    d = {k: float(dimensions[i]) for i, k in enumerate(TLX_DIMENSION_KEYS)}
    if len({round(w, 6) for w in weights}) == 1:
        return analyze_nasa_tlx(d, TLXWeights.uniform()).overall_score
    raise ValueError(
        "Non-uniform TLX weights require pairwise win counts; use "
        "ruxailab_methodology.TLXWeights.from_wins and analyze_nasa_tlx(..., weights=...).",
    )
