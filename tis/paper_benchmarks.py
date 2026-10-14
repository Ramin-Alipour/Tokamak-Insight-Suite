"""Traceable paper-reported benchmarks for the IR-T1 limiter-biasing study.

These are transcriptions of the article's prose, not digitized raw measurements.
Do not merge claims with different limiter positions, bias polarities, or source sections.
"""
from __future__ import annotations

import pandas as pd

P05_SOURCE = {
    "title": "Effects of the location of a biased limiter on turbulent transport in the IR-T1 tokamak plasma",
    "doi": "10.1140/epjd/e2017-80215-6",
    "journal": "European Physical Journal D 71, 228 (2017)",
}

P05_CLAIMS: tuple[dict, ...] = (
    {"claim_id":"P05-A-RADIAL-E","metric":"Radial electric field","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Abstract","reported_change_pct":"increase 35–45%","direction":"increase","lower_pct":35.0,"upper_pct":45.0,"evidence_type":"prose range","audit_status":"conflicts with conclusion range","note":"The abstract reports +35–45%; not a figure-digitized value."},
    {"claim_id":"P05-C-RADIAL-E","metric":"Radial electric field","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Conclusion","reported_change_pct":"increase 20–30%","direction":"increase","lower_pct":20.0,"upper_pct":30.0,"evidence_type":"prose range","audit_status":"conflicts with abstract range","note":"Same stated condition and metric as abstract, but non-overlapping range."},
    {"claim_id":"P05-A-POLOIDAL-TRANSPORT","metric":"Poloidal turbulent transport","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Abstract","reported_change_pct":"increase 25–35%","direction":"increase","lower_pct":25.0,"upper_pct":35.0,"evidence_type":"prose range","audit_status":"compatible with conclusion range","note":"Same range is repeated in conclusion."},
    {"claim_id":"P05-C-POLOIDAL-TRANSPORT","metric":"Poloidal turbulent transport","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Conclusion","reported_change_pct":"increase 25–35%","direction":"increase","lower_pct":25.0,"upper_pct":35.0,"evidence_type":"prose range","audit_status":"compatible with abstract range","note":"Same range is repeated in abstract."},
    {"claim_id":"P05-A-REYNOLDS","metric":"Reynolds-stress parameter","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Abstract","reported_change_pct":"reduction 5–10%","direction":"decrease","lower_pct":5.0,"upper_pct":10.0,"evidence_type":"prose range","audit_status":"broad range; body gives about 5% for described case","note":"Abstract calls this maximum reduction; do not apply to other positions."},
    {"claim_id":"P05-R-REYNOLDS","metric":"Reynolds-stress parameter","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Results §3.6","reported_change_pct":"reduction about 5%","direction":"decrease","lower_pct":5.0,"upper_pct":5.0,"evidence_type":"approximate prose point","audit_status":"consistent with lower end of abstract range","note":"Text reports about 5%; no uncertainty supplied."},
    {"claim_id":"P05-C-POLOIDAL-E","metric":"Poloidal electric field","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Conclusion","reported_change_pct":"decrease 30–40%","direction":"decrease","lower_pct":30.0,"upper_pct":40.0,"evidence_type":"prose range","audit_status":"condition-specific; not a universal bias response","note":"Results text separately says decrease at edge/5 mm but rise nearly 30% at 10 mm under applied bias."},
    {"claim_id":"P05-C-RADIAL-TRANSPORT","metric":"Radial turbulent transport","condition":"Limiter 5 mm inside plasma; +200 V","source_section":"Conclusion","reported_change_pct":"decrease 35–45%","direction":"decrease","lower_pct":35.0,"upper_pct":45.0,"evidence_type":"prose range","audit_status":"approximately compatible with results text","note":"Results §3.5 says reduction is almost 35% for this condition."},
    {"claim_id":"P05-R-EDGE-RADIAL-TRANSPORT","metric":"Radial turbulent transport","condition":"Limiter at plasma edge; +200 V","source_section":"Results §3.5","reported_change_pct":"reduction about 50%","direction":"decrease","lower_pct":50.0,"upper_pct":50.0,"evidence_type":"approximate prose point","audit_status":"different position; must not compare with 5 mm result","note":"Different limiter position from 5 mm benchmark."},
    {"claim_id":"P05-R-POLOIDAL-E-10MM","metric":"Poloidal electric field","condition":"Limiter 10 mm inside plasma; applied ±200 V","source_section":"Results §3.3","reported_change_pct":"increase nearly 30%","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"approximate prose point","audit_status":"different position and bias grouping","note":"Prose defines no statistic or uncertainty for this approximate percentage."},
)


def p05_claims_frame() -> pd.DataFrame:
    """Return a fresh table of source-traceable prose claims."""
    return pd.DataFrame([dict(row) for row in P05_CLAIMS])


def compare_to_p05_range(metric: str, condition: str, observed_change_pct: float) -> dict:
    """Compare a percentage to each numeric prose range for the exact metric/condition."""
    try:
        value = float(observed_change_pct)
    except (TypeError, ValueError) as exc:
        raise ValueError("observed_change_pct must be a finite numeric percentage.") from exc
    if not pd.notna(value):
        raise ValueError("observed_change_pct must be finite.")
    rows = [
        row for row in P05_CLAIMS
        if row["metric"].casefold() == metric.casefold()
        and row["condition"].casefold() == condition.casefold()
        and row["lower_pct"] is not None and row["upper_pct"] is not None
    ]
    if not rows:
        return {"metric":metric,"condition":condition,"observed_change_pct":value,
                "status":"no_numeric_reference_for_exact_condition","matches":[]}
    matches = [
        {"claim_id":row["claim_id"],"source_section":row["source_section"],
         "reported_change_pct":row["reported_change_pct"],
         "within_range":row["lower_pct"] <= abs(value) <= row["upper_pct"],
         "audit_status":row["audit_status"]}
        for row in rows
    ]
    flags=[item["within_range"] for item in matches]
    status=("within_all_reported_ranges" if all(flags) else
            "matches_one_claim_but_conflicts_with_another" if any(flags) else
            "outside_all_reported_ranges")
    return {"metric":metric,"condition":condition,"observed_change_pct":value,
            "status":status,"matches":matches}
