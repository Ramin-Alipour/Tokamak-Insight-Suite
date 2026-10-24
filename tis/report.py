"""Self-contained HTML analysis report (no external assets, safe against markup injection)."""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist()) if v.size <= 200 else {"shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return str(v)


def figure_to_png_base64(fig: Any, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _kv_table(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        val = json.dumps(_json_safe(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple, np.ndarray)) else html.escape(str(v))
        if isinstance(v, (dict, list, tuple, np.ndarray)):
            val = html.escape(val)
        rows.append(f"<tr><th>{html.escape(str(k))}</th><td>{val}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def build_html_report(record: dict[str, Any], sections: list[dict[str, Any]] | None = None,
                      title: str = "Tokamak Insight Suite — Analysis Report") -> str:
    """Build an HTML report.

    record: provenance dict (software, version, project, shot, input hash ...).
    sections: [{"title": str, "results": dict, "figures": [base64_png,...], "notes": str}].
    All text is HTML-escaped; figures are embedded as data URIs.
    """
    sections = sections or []
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1a2a3a}"
             "h1{border-bottom:3px solid #168bff;padding-bottom:.3rem}h2{margin-top:2rem;border-bottom:1px solid #ccd}"
             "table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ccd;padding:4px 8px;text-align:left;vertical-align:top}"
             "th{background:#eef4fb;width:30%}td{word-break:break-word}img{max-width:100%}"
             ".warn{background:#fff6e0;border-left:4px solid #e0a000;padding:.6rem 1rem;margin:1rem 0}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             f"<p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>",
             "<div class='warn'>Research analysis output. Results are valid only for the supplied calibration, units, geometry and "
             "assumptions. This software is not a control, interlock or safety system.</div>",
             "<h2>Provenance</h2>", _kv_table({k: v for k, v in record.items() if k not in {"analysis_history", "data_quality", "diagnostic_maturity_counts", "analysis_maturity_counts"}})]
    diagnostic_maturity = record.get("diagnostic_maturity_counts") or {}
    analysis_maturity = record.get("analysis_maturity_counts") or {}
    if diagnostic_maturity or analysis_maturity:
        parts.append("<h2>Implementation maturity</h2>")
        if diagnostic_maturity:
            parts.append("<h3>Diagnostic categories</h3>")
            parts.append(_kv_table(diagnostic_maturity))
        if analysis_maturity:
            parts.append("<h3>Callable analysis entry points</h3>")
            parts.append(_kv_table(analysis_maturity))
        parts.append("<p>Implementation maturity describes declared software scope only. It is not independent experimental validation or evidence of device-specific calibration.</p>")
    for sec in sections:
        parts.append(f"<h2>{html.escape(str(sec.get('title', 'Result')))}</h2>")
        if sec.get("notes"):
            parts.append(f"<p>{html.escape(str(sec['notes']))}</p>")
        if sec.get("results"):
            parts.append(_kv_table(sec["results"]))
        for png in sec.get("figures", []):
            # The image payload may originate from a caller; escape it as an HTML attribute value.
            safe_png = html.escape(str(png), quote=True)
            parts.append(f"<p><img alt='figure' src='data:image/png;base64,{safe_png}'></p>")
    hist = record.get("analysis_history") or []
    if hist:
        parts.append("<h2>Analysis history</h2><table><tr><th>Export</th><th>Time (UTC)</th></tr>")
        parts += [f"<tr><td>{html.escape(str(h.get('name')))}</td><td>{html.escape(str(h.get('created_at_utc')))}</td></tr>" for h in hist]
        parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts)
