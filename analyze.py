#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.services.explainability import explain
from app.services.nielsen_mapper import map_issue_to_nielsen
from app.services.semantic_analysis import normalize_and_analyze


def _iter_lines(path: Path) -> list[tuple[str, str]]:
    """Return (line_id, text) pairs; skips empty lines."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[str, str]] = []
    for i, line in enumerate(raw.splitlines(), start=1):
        t = line.strip()
        if t:
            out.append((f"{path.name}:{i}", t))
    return out


def run_on_file(
    input_path: Path,
    source: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for rid, text in _iter_lines(input_path):
        rec, analysis = normalize_and_analyze(rid, text, source)
        nielsen = map_issue_to_nielsen(analysis.issue_category)
        exp = explain(analysis, nielsen)
        results.append(
            {
                "record_id": rec.record_id,
                "normalized_text": rec.text,
                "metadata": rec.metadata,
                "analysis": analysis.__dict__,
                "nielsen": (
                    [
                        {"number": h.number, "title": h.title}
                        for h in nielsen.heuristics
                    ]
                    if nielsen
                    else None
                ),
                "explainability": {
                    "summary": exp.summary,
                    "reasoning": exp.reasoning,
                    "recommendation": exp.recommendation,
                },
            },
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RUXAILAB UX insight engine — normalize + semantic analysis (+ explainability).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .txt file or directory of .txt files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--source",
        default="think_aloud",
        choices=["think_aloud", "survey", "moderator_notes", "unknown"],
        help="Normalization profile",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    all_rows: list[dict[str, object]] = []
    if in_path.is_file():
        all_rows.extend(run_on_file(in_path, args.source))
    else:
        for p in sorted(in_path.glob("*.txt")):
            all_rows.extend(run_on_file(p, args.source))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_rows)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
