"""
ingestion.py
============
Reads transcript files from a directory and returns a flat list of records.

Supported formats
-----------------
  *.txt   — line-by-line think-aloud or moderator notes.
            Lines starting with # are metadata/comments.
            Lines containing [MM:SS] are timestamped utterances.
            Lines containing "task answer:" or "observation:" are captured.

  *.csv   — must contain a primary text column.
            Accepted column names (first match wins):
              task_answer · response · text · answer · taskAnswer (demo / legacy)
            Optional columns:
              participant_id, task_id, task_name,
              task_completed, task_time_seconds, expected_time_seconds

Each returned record is a plain dict with at least:
    record_id, source_file, source_type, text, participant_id, task_id, task_name,
    task_completed, task_time_seconds, expected_time_seconds
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional


def ingest_directory(path: Path) -> list[dict]:
    """
    Load transcript data from a directory (all ``*.txt`` / ``*.csv``) or from a
    single ``.txt`` / ``.csv`` file. Same as :func:`ingest_input` (shared implementation).
    """
    return ingest_input(path)


def ingest_input(input_path: Path) -> list[dict]:
    """
    Load from a directory (all ``*.txt`` / ``*.csv``) or a single file.
    Each record includes a stable ``record_id`` (``{file_stem}_{n:04d}``).
    """
    if input_path.is_dir():
        paths = sorted(_list_data_files(input_path))
    elif input_path.is_file():
        if input_path.suffix.lower() not in (".txt", ".csv"):
            raise ValueError(f"Unsupported file type (expected .txt or .csv): {input_path}")
        paths = [input_path]
    else:
        raise FileNotFoundError(input_path)
    return _ingest_paths(paths)


def _list_data_files(directory: Path) -> list[Path]:
    out: list[Path] = []
    for ext in (".txt", ".csv"):
        out.extend(p for p in directory.glob(f"*{ext}") if p.is_file())
    return sorted(out, key=lambda p: p.name.lower())


def _ingest_paths(paths: list[Path]) -> list[dict]:
    all_records: list[dict] = []
    for path in paths:
        chunk = _ingest_file(path)
        for i, r in enumerate(chunk):
            row = dict(r)
            row.setdefault("record_id", f"{path.stem}_{i + 1:04d}")
            all_records.append(row)
    return all_records


def _ingest_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".txt":
        return _ingest_txt(path)
    if path.suffix.lower() == ".csv":
        return _ingest_csv(path)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# TXT ingestion
# ─────────────────────────────────────────────────────────────────────────────

_TIMESTAMP_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}|\d{2}:\d{2})\]\s*(P\d+|M\d+)?:?\s*(.+)"
)
_TASK_ANNO_RE = re.compile(
    r"#\s*Task\s+\d+\s+(?:answer|observation):\s*(.+)", re.IGNORECASE
)
_TASK_ID_RE = re.compile(r"#\s*Task\s+(\d+)", re.IGNORECASE)
_COMPLETED_RE = re.compile(
    r"task.*?(completed|failed|success|could\s*not\s*complete|couldn'?t\s*complete)",
    re.IGNORECASE,
)
_INLINE_ANNO_RE = re.compile(
    r"^(?:task\s*answer|observation)\s*:\s*(.+)\s*$", re.IGNORECASE
)


def _ingest_txt(path: Path) -> list[dict]:
    """
    Parse a think-aloud transcript or moderator notes file.

    Extracts:
      - Timestamped participant utterances  [HH:MM] P01: ...
      - Task annotation comments            # Task N answer: ...
      - Inline task answer / observation:   (no leading #)
      - Plain prose lines (≥ 6 words)       moderator notes / session summaries
    """
    records: list[dict] = []
    source_type = _detect_txt_source_type(path.stem)
    current_task_id: Optional[str] = None

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        task_match = _TASK_ID_RE.search(line)
        if task_match:
            current_task_id = f"task_{int(task_match.group(1))}"

        anno_match = _TASK_ANNO_RE.match(line)
        if anno_match:
            body = anno_match.group(1).strip()
            records.append(
                _make_record(
                    source_file=path.name,
                    source_type=source_type,
                    text=body,
                    task_id=current_task_id,
                    task_completed=_infer_completion(body),
                )
            )
            continue

        if line.startswith("#"):
            continue

        inline = _INLINE_ANNO_RE.match(line)
        if inline:
            body = inline.group(1).strip()
            if body:
                records.append(
                    _make_record(
                        source_file=path.name,
                        source_type=source_type,
                        text=body,
                        task_id=current_task_id,
                        task_completed=_infer_completion(body),
                    )
                )
            continue

        ts_match = _TIMESTAMP_RE.match(line)
        if ts_match:
            speaker = ts_match.group(2) or "unknown"
            text = ts_match.group(3).strip()
            if not text or len(text.split()) < 4:
                continue
            if speaker.startswith("P"):
                records.append(
                    _make_record(
                        source_file=path.name,
                        source_type=source_type,
                        text=text,
                        task_id=current_task_id,
                        participant_id=speaker,
                    )
                )
            continue

        if len(line.split()) >= 6 and not line.startswith("["):
            records.append(
                _make_record(
                    source_file=path.name,
                    source_type=source_type,
                    text=line,
                    task_id=current_task_id,
                    task_completed=_infer_completion(line),
                )
            )

    return records


def _detect_txt_source_type(stem: str) -> str:
    stem_lower = stem.lower()
    if "moderator" in stem_lower or "notes" in stem_lower:
        return "moderator_notes"
    if "think" in stem_lower or "aloud" in stem_lower or "session" in stem_lower:
        return "think_aloud"
    return "think_aloud"


def _infer_completion(text: str) -> Optional[bool]:
    m = _COMPLETED_RE.search(text)
    if not m:
        return None
    g = m.group(1).lower()
    if g in ("completed", "success"):
        return True
    if g == "failed":
        return False
    if "complete" in g:
        return False
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CSV ingestion
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_COLUMN_ALIASES = ("task_answer", "response", "text", "answer", "taskAnswer")


def _ingest_csv(path: Path) -> list[dict]:
    """
    Parse a CSV survey response file.

    Primary text column: first match among aliases in headers.
    """
    records: list[dict] = []

    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        text_col = next((c for c in _TEXT_COLUMN_ALIASES if c in headers), None)

        if text_col is None:
            available = ", ".join(headers)
            raise ValueError(
                f"[ingestion] {path.name}: no recognised text column found. "
                f"Expected one of {_TEXT_COLUMN_ALIASES}. "
                f"Available columns: {available}"
            )

        for row in reader:
            text = row.get(text_col, "").strip()
            if not text or len(text.split()) < 3:
                continue

            completed_raw = row.get("task_completed", "").strip().lower()
            completed: Optional[bool] = None
            if completed_raw in ("true", "yes", "1"):
                completed = True
            elif completed_raw in ("false", "no", "0"):
                completed = False

            rid = (row.get("record_id") or row.get("id") or "").strip() or None

            rec = _make_record(
                source_file=path.name,
                source_type="survey",
                text=text,
                task_id=row.get("task_id", "").strip() or None,
                task_name=row.get("task_name", "").strip() or None,
                participant_id=row.get("participant_id", "").strip() or None,
                task_completed=completed,
                task_time_seconds=_safe_float(row.get("task_time_seconds")),
                expected_time_seconds=_safe_float(row.get("expected_time_seconds")),
            )
            if rid:
                rec["record_id"] = rid
            records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    source_file: str,
    source_type: str,
    text: str,
    task_id: Optional[str] = None,
    task_name: Optional[str] = None,
    participant_id: Optional[str] = None,
    task_completed: Optional[bool] = None,
    task_time_seconds: Optional[float] = None,
    expected_time_seconds: Optional[float] = None,
) -> dict:
    return {
        "source_file": source_file,
        "source_type": source_type,
        "text": text,
        "task_id": task_id,
        "task_name": task_name,
        "participant_id": participant_id,
        "task_completed": task_completed,
        "task_time_seconds": task_time_seconds,
        "expected_time_seconds": expected_time_seconds,
    }


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (ValueError, AttributeError):
        return None
