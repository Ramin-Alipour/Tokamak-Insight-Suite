"""Full-EM CTS identifiability must report actual covariance provenance."""
import pytest

from tokamak_analysis.thomson_identifiability import cts_full_em_identifiability


def _conditional(result, parameter):
    return next(
        item for item in result["conditional"]
        if item["parameter"] == parameter
    )


def test_full_em_identifiability_defaults_external_inputs_to_fixed_not_inferred():
    result = cts_full_em_identifiability(["D+"])
    external = _conditional(result, "external_input_uncertainty_contribution")
    response = _conditional(result, "instrument_response_uncertainty_contribution")
    assert external["status"] == "not_supplied_external_inputs_held_fixed"
    assert response["status"] == "not_supplied_response_held_fixed"
    assert result["non_identifiable"][2]["parameter"] == "magnetic_field_and_mode_polarization"


def test_full_em_identifiability_reports_successful_external_covariance_refits():
    result = cts_full_em_identifiability(
        ["D+", "C6+"],
        external_input_covariance_supplied=True,
        external_input_covariance_propagated=True,
        instrument_response_uncertainty_supplied=True,
        instrument_response_uncertainty_propagated=True,
    )
    external = _conditional(result, "external_input_uncertainty_contribution")
    response = _conditional(result, "instrument_response_uncertainty_contribution")
    assert external["status"].startswith("propagated_by_joint_external_input")
    assert external["covariance_propagated"] is True
    assert response["status"].startswith("propagated_by_measured_response")
    assert response["uncertainty_propagated"] is True
    assert [item["parameter"] for item in result["primary"] if item["parameter"].startswith("ion_temperature")] == [
        "ion_temperature_eV:D+", "ion_temperature_eV:C6+"
    ]


def test_full_em_identifiability_distinguishes_supplied_but_failed_refits():
    result = cts_full_em_identifiability(
        ["D+"],
        external_input_covariance_supplied=True,
        external_input_covariance_propagated=False,
        instrument_response_uncertainty_supplied=True,
        instrument_response_uncertainty_propagated=False,
    )
    assert _conditional(result, "external_input_uncertainty_contribution")["status"] == (
        "supplied_but_not_propagated_insufficient_successful_refits"
    )
    assert _conditional(result, "instrument_response_uncertainty_contribution")["status"] == (
        "supplied_but_not_propagated_insufficient_successful_refits"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"external_input_covariance_propagated": True},
        {"instrument_response_uncertainty_propagated": True},
    ],
)
def test_full_em_identifiability_rejects_impossible_propagation_provenance(kwargs):
    with pytest.raises(ValueError, match="cannot be propagated unless it was supplied"):
        cts_full_em_identifiability(["D+"], **kwargs)
