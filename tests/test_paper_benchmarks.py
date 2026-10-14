import pytest

from tis.paper_benchmarks import P05_CLAIMS, compare_to_p05_range, p05_claims_frame

RADIAL_E_CONDITION = "Limiter 5 mm inside plasma; +200 V"


def test_p05_keeps_abstract_and_conclusion_radial_e_ranges_separate():
    claims = [
        row for row in P05_CLAIMS
        if row["metric"] == "Radial electric field" and row["condition"] == RADIAL_E_CONDITION
    ]
    assert {(row["source_section"], row["lower_pct"], row["upper_pct"]) for row in claims} == {
        ("Abstract", 35.0, 45.0), ("Conclusion", 20.0, 30.0)
    }


def test_p05_comparison_flags_value_matching_only_one_conflicting_claim():
    result = compare_to_p05_range("Radial electric field", RADIAL_E_CONDITION, 25.0)
    assert result["status"] == "matches_one_claim_but_conflicts_with_another"
    assert [item["within_range"] for item in result["matches"]] == [False, True]


def test_p05_comparison_reports_when_value_matches_neither_claim():
    result = compare_to_p05_range("Radial electric field", RADIAL_E_CONDITION, 32.0)
    assert result["status"] == "outside_all_reported_ranges"


def test_p05_exact_metric_and_condition_are_required():
    result = compare_to_p05_range("Poloidal turbulent transport", RADIAL_E_CONDITION, 30.0)
    assert result["status"] == "no_numeric_reference_for_exact_condition"


def test_p05_claim_table_is_exportable_and_retains_provenance_fields():
    frame = p05_claims_frame()
    assert len(frame) == len(P05_CLAIMS)
    assert {"claim_id", "source_section", "condition", "reported_change_pct", "audit_status"} <= set(frame.columns)
    assert frame["claim_id"].is_unique


def test_p05_rejects_non_numeric_comparison():
    with pytest.raises(ValueError):
        compare_to_p05_range("Radial electric field", RADIAL_E_CONDITION, float("nan"))
