from __future__ import annotations

from dataclasses import dataclass

from ruxailab_methodology import HeuristicMatch, map_to_heuristics


@dataclass(frozen=True)
class NielsenHeuristic:
    number: int
    title: str


@dataclass(frozen=True)
class NielsenMapping:
    heuristics: tuple[NielsenHeuristic, ...]


def _match_to_mapping(m: HeuristicMatch) -> NielsenMapping | None:
    if m.primary is None:
        return None
    out: list[NielsenHeuristic] = [
        NielsenHeuristic(m.primary_number or 0, m.primary_name or ""),
    ]
    if m.secondary_number:
        out.append(NielsenHeuristic(m.secondary_number, m.secondary_name or ""))
    return NielsenMapping(tuple(out))


def map_issue_to_nielsen(issue_category: str | None) -> NielsenMapping | None:
    if not issue_category:
        return None
    return _match_to_mapping(map_to_heuristics(issue_category))
