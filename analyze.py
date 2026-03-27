#!/usr/bin/env python3
"""
analyze.py
==========
CLI entry point for the RUXAILAB usability analysis pipeline.

Usage
-----
    python analyze.py --input transcripts/ --output output/report.html

    # Full options:
    python analyze.py \\
        --input  transcripts/ \\
        --output output/report.html \\
        --method nielsen \\
        --min-confidence 0.2 \\
        --title  "Q4 Usability Study" \\
        --verbose

Pipeline stages
---------------
  1. Ingest    — read .txt and .csv files via ingestion.ingest_directory
  2. Normalize — clean text via ruxailab_nlp.normalize_text
  3. Analyze   — classify issues via SemanticAnalyzer
  4. Explain   — build reasoning chains via ExplainabilityEngine
  5. Aggregate — roll up to study-level report
  6. Render    — write HTML report via report_generator.generate_html_report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from explainability import ExplainedResult


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="RUXAILAB usability analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --input transcripts/ --output report.html
  python analyze.py --input transcripts/ --output report.html --verbose
  python analyze.py --input transcripts/ --output report.html --min-confidence 0.3
        """,
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing .txt/.csv files, or a single file.",
    )
    p.add_argument(
        "--output",
        "-o",
        default="output/report.html",
        help="Output HTML report path (default: output/report.html).",
    )
    p.add_argument(
        "--method",
        "-m",
        default="nielsen",
        choices=["nielsen", "heuristic", "all"],
        help="Analysis method label shown in the report (default: nielsen).",
    )
    p.add_argument(
        "--title",
        "-t",
        default="RUXAILAB Usability Study",
        help="Study title shown in the report header.",
    )
    p.add_argument(
        "--study-id",
        default="",
        help="Optional study ID for report metadata.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="Minimum confidence to include a finding (default: 0.15).",
    )
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Omit reasoning chains from the HTML report.",
    )
    p.add_argument(
        "--no-raw-text",
        action="store_true",
        help="Omit raw input text from the HTML report.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-finding details to stdout.",
    )
    return p.parse_args()


def _print_header(args: argparse.Namespace) -> None:
    print()
    print("=" * 60)
    print("  RUXAILAB Usability Analysis Pipeline")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Method  : {args.method}")
    print(f"  Min conf: {args.min_confidence}")
    print("=" * 60)
    print()


def _print_finding(explained: "ExplainedResult") -> None:
    a = explained.analysis
    print(
        f"  [{a.severity.value:<6}] {a.issue_category.value:<20} "
        f"conf={a.confidence:.2f}  {a.raw_text[:60]!r}"
    )


def _print_summary(report_data: object) -> None:
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total analyzed    : {report_data.total_analyzed}")
    print(f"  Issues found      : {report_data.usability_issue_count}")
    print(f"  High priority     : {report_data.high_priority_count}")
    print(f"  Severity dist.    : {report_data.severity_distribution}")
    print(f"  Category dist.    : {report_data.category_distribution}")
    print(f"  Sentiment         : {report_data.overall_sentiment_ratio}")
    if report_data.heuristic_frequency:
        top_h = max(report_data.heuristic_frequency, key=report_data.heuristic_frequency.get)
        print(
            f"  Most-hit heuristic: H{top_h} "
            f"({report_data.heuristic_frequency[top_h]} finding(s))"
        )
    print("=" * 60)
    print()


def main() -> int:
    args = _parse_args()

    try:
        from ingestion import ingest_directory
        from ruxailab_nlp import TextSource, get_default_config_for_source, normalize_text
        from explainability import (
            ExplainabilityEngine,
            SemanticAnalyzer,
            StudyExplainabilityAggregator,
            TaskContext,
        )
        from report_generator import generate_html_report
    except ImportError as exc:
        print(f"[ERROR] Missing module: {exc}", file=sys.stderr)
        print("Make sure all pipeline files are in the same directory.", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    _print_header(args)

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    print("Stage 1/6  Ingesting transcripts…")
    try:
        raw_records = ingest_directory(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    by_file: dict[str, int] = {}
    for r in raw_records:
        fn = r.get("source_file", "")
        by_file[fn] = by_file.get(fn, 0) + 1
    for name in sorted(by_file):
        print(f"           {name:<40} {by_file[name]:>3} records")
    print(f"\n           {len(raw_records)} total records loaded\n")

    if not raw_records:
        print("[WARN] No records found. Check --input contains .txt or .csv files.")
        return 0

    # ── Stage 2: Normalize ────────────────────────────────────────────────────
    print("Stage 2/6  Normalizing text…")

    _source_map = {
        "think_aloud": TextSource.TRANSCRIPTION,
        "survey": TextSource.TASK_ANSWER,
        "moderator_notes": TextSource.STUDY_TEXT,
        "moderator": TextSource.STUDY_TEXT,
        "study_text": TextSource.STUDY_TEXT,
    }

    normalized: list[dict] = []
    for rec in raw_records:
        source_enum = _source_map.get(rec["source_type"], TextSource.TASK_ANSWER)
        cfg = get_default_config_for_source(source_enum)
        norm_result = normalize_text(rec["text"], cfg)
        normalized.append(
            {
                **rec,
                "normalized": norm_result.normalized_text,
                "language": norm_result.language_hint,
            }
        )

    print(f"           {len(normalized)} records normalized\n")

    # ── Stage 3 + 4: Analyze & Explain ───────────────────────────────────────
    print("Stage 3/6  Running semantic analysis…")
    print("Stage 4/6  Building explainability chains…")

    analyzer = SemanticAnalyzer()
    engine = ExplainabilityEngine()
    explained_records: list[dict] = []
    skipped_low_conf = 0

    for rec in normalized:
        text = rec["normalized"]
        if not text or len(text.split()) < 3:
            continue

        ctx = TaskContext(
            task_id=rec.get("task_id"),
            task_name=rec.get("task_name"),
            task_completed=rec.get("task_completed"),
            task_time_seconds=rec.get("task_time_seconds"),
            expected_time_seconds=rec.get("expected_time_seconds"),
            source_field=str(rec.get("source_type", "unknown")),
            respondent_id=rec.get("participant_id"),
        )

        analysis = analyzer.analyze(text, ctx)

        if analysis.confidence < args.min_confidence:
            skipped_low_conf += 1
            continue

        explained = engine.explain(analysis)
        explained_records.append({**rec, "explained": explained})

        if args.verbose:
            _print_finding(explained)

    kept = len(explained_records)
    print(
        f"           {kept} findings kept  "
        f"({skipped_low_conf} skipped below conf={args.min_confidence})\n"
    )

    # ── Stage 5: Aggregate ────────────────────────────────────────────────────
    print("Stage 5/6  Aggregating study report…")
    all_explained = [r["explained"] for r in explained_records]
    report_data = StudyExplainabilityAggregator().aggregate(all_explained)
    print(
        f"           {report_data.usability_issue_count} usability issues | "
        f"{report_data.high_priority_count} high priority\n"
    )

    # ── Stage 6: Render ───────────────────────────────────────────────────────
    print(f"Stage 6/6  Rendering HTML report → {output_path}")

    total_files = len({r["source_file"] for r in raw_records})
    generate_html_report(
        report_data,
        explained_records,
        output_path,
        {
            "input_dir": str(input_path),
            "method": args.method,
            "min_confidence": args.min_confidence,
            "total_files": total_files,
        },
        title=args.title,
        study_id=args.study_id,
        show_raw_text=not args.no_raw_text,
        show_reasoning=not args.no_reasoning,
    )

    size_kb = output_path.stat().st_size // 1024
    elapsed = time.perf_counter() - t_start

    print(f"\nDone in {elapsed:.1f}s  —  {output_path}  ({size_kb} KB)\n")
    _print_summary(report_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
