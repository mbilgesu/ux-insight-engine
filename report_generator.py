"""
report_generator.py
===================
Renders the explainability pipeline output to a self-contained HTML report.

The report is designed to be:
  - Self-contained (no CDN dependencies — works offline)
  - Pasteable into Study.studyConclusion (plain-text reasoning chains)
  - Printable (CSS print media query included)
  - RUXAILAB-branded (matches the orange/blue palette)

Called by analyze.py after StudyExplainabilityAggregator.aggregate().

Public interface
----------------
  generate_html_report(
      report_data,
      explained_records,
      output_path,
      meta,
      title=...,
      study_id=...,
      show_raw_text=...,
      show_reasoning=...,
  ) -> None
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_html_report(
    report_data: Any,
    explained_records: list[dict],
    output_path: Path,
    meta: dict,
    title: str = "RUXAILAB Usability Study",
    study_id: str = "",
    show_raw_text: bool = True,
    show_reasoning: bool = True,
) -> None:
    """Write the full HTML report to ``output_path``."""
    merged: dict[str, Any] = {
        **meta,
        "show_raw_text": show_raw_text,
        "show_reasoning": show_reasoning,
    }
    generated_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = _build_body(
        report_data,
        explained_records,
        merged,
        title=title,
        study_id=study_id,
        generated_ts=generated_ts,
    )
    full = (
        _HTML_SHELL.replace("{{BODY}}", body)
        .replace("{{GENERATED}}", html.escape(generated_ts))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Body builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_body(
    report_data: Any,
    explained_records: list[dict],
    meta: dict,
    *,
    title: str,
    study_id: str,
    generated_ts: str,
) -> str:
    parts = [
        _section_header(
            meta,
            title=title,
            study_id=study_id,
            generated_ts=generated_ts,
        ),
        _section_kpi_row(report_data),
        _section_heuristic_chart(report_data),
        _section_top_issues(report_data),
        _section_all_findings(explained_records, meta),
        _section_recommendations(report_data),
        _section_reasoning_appendix(explained_records, meta),
    ]
    return "\n".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────


def _section_header(
    meta: dict,
    *,
    title: str,
    study_id: str,
    generated_ts: str,
) -> str:
    method = html.escape(str(meta.get("method", "nielsen")).upper())
    idir = html.escape(str(meta.get("input_dir", "")))
    files = meta.get("total_files", "?")
    min_c = meta.get("min_confidence")
    min_bit = (
        f' &nbsp;·&nbsp; Min confidence: <strong>{html.escape(str(min_c))}</strong>'
        if min_c is not None
        else ""
    )
    title_esc = html.escape(title)
    sid = str(study_id or "").strip()
    sid_bit = (
        f' &nbsp;·&nbsp; Study ID: <code>{html.escape(sid)}</code>'
        if sid
        else ""
    )
    ts_esc = html.escape(generated_ts)
    return f"""
<header class="report-header">
  <div class="logo-row">
    <span class="logo-r">R</span>
    <span class="logo-text">UXAILAB</span>
    <span class="report-badge">Usability Analysis Report</span>
  </div>
  <h1>{title_esc}</h1>
  <p class="subtitle">
    Method: <strong>{method}</strong> &nbsp;·&nbsp;
    Source: <code>{idir}</code> ({files} file(s)){sid_bit}{min_bit} &nbsp;·&nbsp;
    Generated: <span id="ts">{ts_esc}</span>
  </p>
</header>
"""


def _section_kpi_row(report_data: Any) -> str:
    total = report_data.total_analyzed
    issues = report_data.usability_issue_count
    high_pri = report_data.high_priority_count
    sev = report_data.severity_distribution
    hi = sev.get("HIGH", 0)
    med = sev.get("MEDIUM", 0)
    low = sev.get("LOW", 0)
    sent = report_data.overall_sentiment_ratio
    neg_pct = round(sent.get("negative", 0) / max(total, 1) * 100)

    return f"""
<section class="kpi-row">
  <div class="kpi kpi-blue">
    <div class="kpi-value">{total}</div>
    <div class="kpi-label">Responses Analyzed</div>
  </div>
  <div class="kpi kpi-orange">
    <div class="kpi-value">{issues}</div>
    <div class="kpi-label">Usability Issues</div>
  </div>
  <div class="kpi kpi-red">
    <div class="kpi-value">{high_pri}</div>
    <div class="kpi-label">High Priority</div>
  </div>
  <div class="kpi kpi-sev">
    <div class="kpi-value">
      <span class="badge-high">{hi}H</span>
      <span class="badge-med">{med}M</span>
      <span class="badge-low">{low}L</span>
    </div>
    <div class="kpi-label">Severity Breakdown</div>
  </div>
  <div class="kpi kpi-neg">
    <div class="kpi-value">{neg_pct}%</div>
    <div class="kpi-label">Negative Sentiment</div>
  </div>
</section>
"""


def _section_heuristic_chart(report_data: Any) -> str:
    hf = report_data.heuristic_frequency
    if not hf:
        return ""

    _HEURISTIC_NAMES = {
        1: "H1 Visibility of Status",
        2: "H2 Match Real World",
        3: "H3 User Control",
        4: "H4 Consistency",
        5: "H5 Error Prevention",
        6: "H6 Recognition",
        7: "H7 Flexibility",
        8: "H8 Aesthetics",
        9: "H9 Error Recovery",
        10: "H10 Help & Docs",
    }

    max_count = max(hf.values()) or 1
    bars = ""
    for h_num in sorted(hf.keys()):
        count = hf[h_num]
        pct = round(count / max_count * 100)
        name = _HEURISTIC_NAMES.get(h_num, f"H{h_num}")
        color = "#e65100" if pct == 100 else ("#f57c00" if pct >= 60 else "#1976d2")
        bars += f"""
    <div class="hbar-row">
      <span class="hbar-label">{html.escape(name)}</span>
      <div class="hbar-track">
        <div class="hbar-fill" style="width:{pct}%; background:{color}"></div>
      </div>
      <span class="hbar-count">{count}</span>
    </div>"""

    return f"""
<section class="card">
  <h2>Nielsen Heuristic Frequency</h2>
  <p class="card-sub">How many findings map to each heuristic — higher bar = more affected.</p>
  <div class="hbar-chart">{bars}
  </div>
</section>
"""


def _section_top_issues(report_data: Any) -> str:
    if not report_data.top_issues:
        return ""

    cards = ""
    for i, ex in enumerate(report_data.top_issues, 1):
        a = ex.analysis
        sev = a.severity.value
        cat = a.issue_category.value.replace("_", " ").title()
        conf = round(a.confidence * 100)
        nref = html.escape(ex.nielsen_ref or "—")
        rec = html.escape(
            ex.recommendation[:160] + ("…" if len(ex.recommendation) > 160 else "")
        )
        text = html.escape(
            a.raw_text[:120] + ("…" if len(a.raw_text) > 120 else "")
        )
        sev_class = {"HIGH": "sev-high", "MEDIUM": "sev-med", "LOW": "sev-low"}.get(
            sev, "sev-none"
        )
        hp_badge = (
            '<span class="hp-badge">High priority</span>' if ex.is_high_priority else ""
        )

        cards += f"""
  <div class="issue-card {sev_class}">
    <div class="issue-card-header">
      <span class="issue-num">#{i}</span>
      <span class="issue-cat">{cat}</span>
      <span class="issue-sev sev-badge-{sev.lower()}">{sev}</span>
      {hp_badge}
      <span class="issue-conf">{conf}% confidence</span>
    </div>
    <blockquote class="issue-quote">{text}</blockquote>
    <div class="issue-heuristic">{nref}</div>
    <div class="issue-rec">{rec}</div>
  </div>"""

    return f"""
<section class="card">
  <h2>Top Findings</h2>
  <p class="card-sub">Highest-severity issues sorted by confidence.</p>
  {cards}
</section>
"""


def _section_all_findings(explained_records: list[dict], meta: dict) -> str:
    if not explained_records:
        return ""

    show_raw = bool(meta.get("show_raw_text", True))
    rows = ""
    for rec in explained_records:
        ex = rec["explained"]
        a = ex.analysis
        sev = a.severity.value
        cat = a.issue_category.value.replace("_", " ").title()
        pid = html.escape(str(rec.get("participant_id") or "—"))
        src = html.escape(str(rec.get("source_type", "—")))
        tid = html.escape(str(rec.get("task_id") or "—"))
        raw_ingest = rec.get("text") or ""
        if show_raw and raw_ingest:
            snippet = raw_ingest[:90] + ("…" if len(raw_ingest) > 90 else "")
            norm_note = a.raw_text[:90] + ("…" if len(a.raw_text) > 90 else "")
            txt = html.escape(snippet)
            norm_esc = html.escape(norm_note)
            txt_cell = (
                f'<div class="quote-cell"><span class="raw-label">Raw</span> {txt}<br/>'
                f'<span class="raw-label">Normalized</span> {norm_esc}</div>'
            )
        else:
            norm_note = a.raw_text[:90] + ("…" if len(a.raw_text) > 90 else "")
            txt_cell = f'<div class="quote-cell">{html.escape(norm_note)}</div>'
        conf = round(a.confidence * 100)
        nref = html.escape(ex.nielsen_ref or "—")
        sev_badge = f'<span class="sev-badge-{sev.lower()}">{sev}</span>'
        hp = "*" if ex.is_high_priority else ""

        rows += f"""
    <tr>
      <td>{sev_badge} {hp}</td>
      <td>{cat}</td>
      <td class="mono">{pid}</td>
      <td class="mono">{src}</td>
      <td class="mono">{tid}</td>
      <td>{txt_cell}</td>
      <td>{conf}%</td>
      <td class="small-text">{nref}</td>
    </tr>"""

    return f"""
<section class="card">
  <h2>All Findings</h2>
  <div class="table-scroll">
  <table class="findings-table">
    <thead>
      <tr>
        <th>Severity</th><th>Category</th><th>Participant</th>
        <th>Source</th><th>Task</th><th>Text</th>
        <th>Conf.</th><th>Heuristic</th>
      </tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>
  </div>
</section>
"""


def _section_recommendations(report_data: Any) -> str:
    recs = report_data.recommendations_deduplicated
    if not recs:
        return ""

    items = "".join(f'<li class="rec-item">{html.escape(r)}</li>' for r in recs[:10])
    return f"""
<section class="card">
  <h2>Design Recommendations</h2>
  <p class="card-sub">Ordered by frequency of supporting evidence across participants.</p>
  <ol class="rec-list">{items}</ol>
</section>
"""


def _section_reasoning_appendix(explained_records: list[dict], meta: dict) -> str:
    if not explained_records or not meta.get("show_reasoning", True):
        return ""

    blocks = ""
    for rec in explained_records:
        ex = rec["explained"]
        a = ex.analysis
        pid = rec.get("participant_id") or "unknown"
        src = rec.get("source_type", "")
        cat = a.issue_category.value
        sev = a.severity.value

        chain_html = "".join(
            f'<div class="chain-step">{html.escape(step)}</div>'
            for step in ex.reasoning_chain
        )
        ev_html = "".join(f"<li>{html.escape(e)}</li>" for e in ex.evidence)

        rec_line = (
            f'<div class="rec-block">{html.escape(ex.recommendation)}</div>'
            if ex.recommendation
            else ""
        )

        blocks += f"""
  <details class="chain-block">
    <summary>
      <span class="sev-badge-{sev.lower()}">{sev}</span>
      <strong>{html.escape(cat)}</strong> —
      {html.escape(str(pid))} / {html.escape(str(src))}
    </summary>
    <div class="chain-body">
      <h4>Evidence</h4>
      <ul class="evidence-list">{ev_html}</ul>
      <h4>Reasoning Chain</h4>
      <div class="chain-steps">{chain_html}</div>
      {rec_line}
    </div>
  </details>"""

    return f"""
<section class="card">
  <h2>Explainability Appendix — Full Reasoning Chains</h2>
  <p class="card-sub">
    Every classification decision is fully traceable.
    Click any finding to expand its step-by-step reasoning chain.
  </p>
  {blocks}
</section>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML shell with embedded CSS
# ─────────────────────────────────────────────────────────────────────────────

_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RUXAILAB — Usability Analysis Report</title>
<style>
/* ── Reset & Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f7ff; color: #212121; line-height: 1.6; font-size: 14px; }
a { color: #1976d2; }

/* ── Header ─────────────────────────────────────────────────── */
.report-header { background: linear-gradient(135deg, #0d47a1 0%, #1565c0 60%, #e65100 100%);
  color: #fff; padding: 2rem 2.5rem 1.5rem; }
.logo-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .75rem; }
.logo-r   { font-size: 2rem; font-weight: 900; color: #ff6d00; background: #fff;
            border-radius: 6px; padding: 0 8px; line-height: 1.3; }
.logo-text { font-size: 1.4rem; font-weight: 700; letter-spacing: .5px; }
.report-badge { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.4);
                border-radius: 20px; padding: 2px 12px; font-size: .75rem; margin-left: .5rem; }
.report-header h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: .4rem; }
.subtitle { opacity: .85; font-size: .85rem; }
.subtitle code { background: rgba(255,255,255,.2); border-radius: 4px; padding: 1px 6px; }

/* ── KPI Row ────────────────────────────────────────────────── */
.kpi-row { display: flex; flex-wrap: wrap; gap: 1rem; padding: 1.5rem 2rem; }
.kpi { background: #fff; border-radius: 12px; padding: 1.2rem 1.5rem;
       flex: 1; min-width: 140px; box-shadow: 0 2px 8px rgba(0,0,0,.07); }
.kpi-value { font-size: 2rem; font-weight: 800; line-height: 1; }
.kpi-label { font-size: .75rem; color: #757575; margin-top: .3rem; }
.kpi-blue   .kpi-value { color: #1565c0; }
.kpi-orange .kpi-value { color: #e65100; }
.kpi-red    .kpi-value { color: #c62828; }
.kpi-neg    .kpi-value { color: #6a1b9a; }
.badge-high { color: #c62828; font-size: 1.1rem; font-weight: 700; margin-right: .25rem; }
.badge-med  { color: #e65100; font-size: 1.1rem; font-weight: 700; margin-right: .25rem; }
.badge-low  { color: #2e7d32; font-size: 1.1rem; font-weight: 700; }

/* ── Cards ──────────────────────────────────────────────────── */
.card { background: #fff; border-radius: 12px; padding: 1.5rem 2rem;
        margin: 0 2rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.07); }
.card h2 { font-size: 1.1rem; font-weight: 700; margin-bottom: .3rem; color: #1565c0; }
.card-sub { font-size: .8rem; color: #757575; margin-bottom: 1rem; }

/* ── Heuristic Bar Chart ────────────────────────────────────── */
.hbar-chart { display: flex; flex-direction: column; gap: .5rem; margin-top: .75rem; }
.hbar-row   { display: flex; align-items: center; gap: .75rem; }
.hbar-label { width: 190px; font-size: .78rem; flex-shrink: 0; color: #424242; }
.hbar-track { flex: 1; background: #e8eaf6; border-radius: 4px; height: 18px; overflow: hidden; }
.hbar-fill  { height: 100%; border-radius: 4px; transition: width .3s; }
.hbar-count { width: 24px; text-align: right; font-size: .8rem; font-weight: 600; color: #424242; }

/* ── Issue Cards ────────────────────────────────────────────── */
.issue-card { border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: .75rem;
              border-left: 4px solid #9e9e9e; background: #fafafa; }
.sev-high   { border-left-color: #c62828; background: #fff8f8; }
.sev-med    { border-left-color: #e65100; background: #fffaf5; }
.sev-low    { border-left-color: #2e7d32; background: #f8fff8; }
.issue-card-header { display: flex; flex-wrap: wrap; align-items: center;
                     gap: .5rem; margin-bottom: .5rem; }
.issue-num  { font-weight: 800; font-size: 1rem; color: #757575; }
.issue-cat  { font-weight: 700; font-size: .9rem; }
.issue-conf { font-size: .75rem; color: #9e9e9e; margin-left: auto; }
.sev-badge-high   { background: #ffebee; color: #c62828; border-radius: 4px;
                    padding: 1px 8px; font-size: .75rem; font-weight: 700; }
.sev-badge-medium,
.sev-badge-med    { background: #fff3e0; color: #e65100; border-radius: 4px;
                    padding: 1px 8px; font-size: .75rem; font-weight: 700; }
.sev-badge-low    { background: #e8f5e9; color: #2e7d32; border-radius: 4px;
                    padding: 1px 8px; font-size: .75rem; font-weight: 700; }
.sev-badge-none   { background: #f5f5f5; color: #757575; border-radius: 4px;
                    padding: 1px 8px; font-size: .75rem; font-weight: 700; }
.hp-badge   { background: #fff8e1; color: #f57f17; border-radius: 4px;
              padding: 1px 8px; font-size: .73rem; font-weight: 700; }
.issue-quote { font-style: italic; color: #616161; border-left: 2px solid #e0e0e0;
               padding-left: .6rem; margin: .4rem 0; font-size: .85rem; }
.issue-heuristic { font-size: .82rem; color: #1565c0; margin-top: .35rem; }
.issue-rec  { font-size: .82rem; color: #424242; margin-top: .3rem; }

/* ── Findings Table ─────────────────────────────────────────── */
.table-scroll { overflow-x: auto; }
.findings-table { width: 100%; border-collapse: collapse; font-size: .8rem; }
.findings-table th { background: #e8eaf6; color: #1565c0; padding: .5rem .75rem;
                     text-align: left; font-weight: 700; white-space: nowrap; }
.findings-table td { padding: .45rem .75rem; border-bottom: 1px solid #f0f0f0;
                     vertical-align: top; }
.findings-table tr:hover td { background: #fafafa; }
.mono       { font-family: monospace; font-size: .78rem; }
.small-text { font-size: .75rem; color: #616161; }
.quote-cell { max-width: 280px; color: #424242; }
.raw-label  { font-size: .7rem; color: #9e9e9e; text-transform: uppercase; letter-spacing: .04em; }

/* ── Recommendations ────────────────────────────────────────── */
.rec-list { padding-left: 1.2rem; }
.rec-item { padding: .4rem 0; border-bottom: 1px solid #f0f0f0; font-size: .88rem; }
.rec-item:last-child { border-bottom: none; }

/* ── Reasoning Chain Appendix ───────────────────────────────── */
.chain-block { border: 1px solid #e0e0e0; border-radius: 8px;
               margin-bottom: .6rem; overflow: hidden; }
.chain-block summary { padding: .65rem 1rem; cursor: pointer; font-size: .85rem;
                       background: #fafafa; display: flex; align-items: center;
                       gap: .5rem; list-style: none; }
.chain-block summary:hover { background: #f0f4ff; }
.chain-block summary::-webkit-details-marker { display: none; }
.chain-block[open] summary { background: #e8eaf6; }
.chain-body { padding: 1rem 1.2rem; border-top: 1px solid #e0e0e0; }
.chain-body h4 { font-size: .8rem; font-weight: 700; color: #757575;
                 text-transform: uppercase; letter-spacing: .5px; margin: .6rem 0 .3rem; }
.chain-body h4:first-child { margin-top: 0; }
.evidence-list { padding-left: 1rem; font-size: .82rem; color: #424242; }
.evidence-list li { margin-bottom: .2rem; }
.chain-steps { display: flex; flex-direction: column; gap: .25rem; }
.chain-step  { font-size: .8rem; font-family: monospace; background: #f5f5f5;
               border-radius: 4px; padding: .3rem .6rem; color: #212121; white-space: pre-wrap; }
.rec-block   { margin-top: .75rem; padding: .6rem .8rem; background: #fff8e1;
               border-radius: 6px; font-size: .83rem; color: #424242; }

/* ── Print ──────────────────────────────────────────────────── */
@media print {
  .card { box-shadow: none; border: 1px solid #e0e0e0; }
  .chain-block[open] summary { background: #f0f0f0; }
  details { display: block; }
  details summary { display: none; }
}

/* ── Footer ─────────────────────────────────────────────────── */
footer { text-align: center; padding: 2rem; color: #9e9e9e; font-size: .78rem; }
</style>
</head>
<body>

{{BODY}}

<footer>
  RUXAILAB Usability Analysis Pipeline &nbsp;·&nbsp;
  Generated {{GENERATED}} &nbsp;·&nbsp;
  All findings are AI-assisted; manual expert review is recommended for HIGH severity issues.
</footer>

</body>
</html>
"""
