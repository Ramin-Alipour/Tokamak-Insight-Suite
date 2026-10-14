"""Traceable paper-reported benchmarks for the IR-T1 limiter-biasing study.

These are transcriptions of the article's prose, not digitized raw measurements.
Do not merge claims with different limiter positions, bias polarities, or source sections.
"""
from __future__ import annotations

import math
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
    {"claim_id":"P05-R-IP-EDGE-POS","metric":"Plasma current","condition":"Limiter at plasma edge; +200 V","source_section":"Results §3.1","reported_change_pct":"current improves and lasts slightly longer","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"qualitative; no percentage benchmark","note":"Compared with the typical discharge/no applied bias described in the text."},
    {"claim_id":"P05-R-IP-EDGE-NEG","metric":"Plasma current","condition":"Limiter at plasma edge; −200 V","source_section":"Results §3.1","reported_change_pct":"current rate drops significantly","direction":"decrease","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"qualitative; no percentage benchmark","note":"The prose gives no uncertainty or common quantitative comparison statistic."},
    {"claim_id":"P05-R-IP-5MM","metric":"Plasma current","condition":"Limiter 5 mm inside plasma; ±200 V","source_section":"Results §3.1","reported_change_pct":"−200 V: discharge lasts up to 31 ms; +200 V: earlier disruption","direction":"mixed","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"polarity-dependent result","note":"The text also says the maximum current exceeds 20 kA at the 5 mm position, without assigning that maximum to a single bias polarity."},
    {"claim_id":"P05-R-IP-10MM","metric":"Plasma current","condition":"Limiter 10 mm inside plasma; ±200 V","source_section":"Results §3.1","reported_change_pct":"worst current behavior; more fluctuations and instabilities","direction":"mixed","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"condition-specific instability finding","note":"Do not convert the qualitative description into a numerical reduction."},
    {"claim_id":"P05-R-RADIAL-E-EDGE","metric":"Radial electric field","condition":"Limiter at plasma edge; +200 V","source_section":"Results §3.2","reported_change_pct":"largest average increase at the edge in the described comparison","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"qualitative; no numeric range in results prose","note":"The abstract's numeric range is associated especially with the 5 mm position, not this edge-only comparison."},
    {"claim_id":"P05-R-RADIAL-E-5MM","metric":"Radial electric field","condition":"Limiter 5 mm inside plasma; ±200 V","source_section":"Results §3.2","reported_change_pct":"both positive and negative bias raise the field noticeably","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"qualitative; does not resolve abstract/conclusion range conflict","note":"The numeric prose conflict is separately retained in the abstract and conclusion records."},
    {"claim_id":"P05-R-RADIAL-E-10MM","metric":"Radial electric field","condition":"Limiter 10 mm inside plasma; +200 V versus −200 V","source_section":"Results §3.2","reported_change_pct":"+200 V improves the field more than −200 V; −200 V also has an effect","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"condition- and polarity-specific","note":"No quantitative percentage is given in the results prose for this comparison."},
    {"claim_id":"P05-R-POLOIDAL-E-EDGE","metric":"Poloidal electric field","condition":"Limiter at plasma edge; +200 V","source_section":"Results §3.3","reported_change_pct":"field reaches its lowest rate in the described comparison","direction":"decrease","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"qualitative; separate from 5 mm conclusion range","note":"Do not apply the conclusion's 30–40% reduction to the edge condition."},
    {"claim_id":"P05-R-POLOIDAL-TRANSPORT-EDGE","metric":"Poloidal turbulent transport","condition":"Limiter at plasma edge; ±200 V","source_section":"Results §3.4","reported_change_pct":"+200 V increases; −200 V decreases relative to no applied voltage","direction":"mixed","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"polarity-dependent result","note":"The abstract/conclusion 25–35% range refers especially to the 5 mm / +200 V condition."},
    {"claim_id":"P05-R-POLOIDAL-TRANSPORT-5MM","metric":"Poloidal turbulent transport","condition":"Limiter 5 mm inside plasma; ±200 V","source_section":"Results §3.4","reported_change_pct":"both polarities increase transport; +200 V has the larger effect","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"qualitative; consistent with reported +25–35% claim for +200 V","note":"The negative-bias increase is not assigned a numerical percentage in the text."},
    {"claim_id":"P05-R-POLOIDAL-TRANSPORT-10MM","metric":"Poloidal turbulent transport","condition":"Limiter 10 mm inside plasma; ±200 V","source_section":"Results §3.4","reported_change_pct":"−200 V produces the largest increase; +200 V has more fluctuations","direction":"mixed","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"position-dependent behavior","note":"Do not generalize the +200 V, 5 mm benchmark to 10 mm."},
    {"claim_id":"P05-R-RADIAL-TRANSPORT-5MM-NEG","metric":"Radial turbulent transport","condition":"Limiter 5 mm inside plasma; −200 V","source_section":"Results §3.5","reported_change_pct":"negative bias is not effective at reducing transport","direction":"unchanged_or_increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"polarity-specific result","note":"The nearly 35% reduction is reported for +200 V, not −200 V."},
    {"claim_id":"P05-R-RADIAL-TRANSPORT-10MM","metric":"Radial turbulent transport","condition":"Limiter 10 mm inside plasma; ±200 V","source_section":"Results §3.5","reported_change_pct":"transport increases under both applied polarities","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"position-dependent behavior","note":"No percentage is provided for this condition in the prose."},
    {"claim_id":"P05-R-REYNOLDS-EDGE","metric":"Reynolds-stress parameter","condition":"Limiter at plasma edge; ±200 V","source_section":"Results §3.6","reported_change_pct":"+200 V reduces by 15%; −200 V increases it","direction":"mixed","lower_pct":None,"upper_pct":None,"evidence_type":"prose numeric point plus qualitative contrast","audit_status":"different position from 5 mm benchmark","note":"The 15% reduction at the edge must not be confused with the 5–10% maximum reduction reported for 5 mm."},
    {"claim_id":"P05-R-REYNOLDS-10MM","metric":"Reynolds-stress parameter","condition":"Limiter 10 mm inside plasma; ±200 V","source_section":"Results §3.6","reported_change_pct":"large fluctuations and dramatic increase under applied bias","direction":"increase","lower_pct":None,"upper_pct":None,"evidence_type":"qualitative prose","audit_status":"condition-specific instability finding","note":"The text provides no numerical range for this condition."},
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
    if not math.isfinite(value):
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
