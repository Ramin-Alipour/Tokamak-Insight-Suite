import numpy as np
import pytest

from tokamak_analysis.thomson_contracts import (
    compose_spectral_covariance,
    diagnose_local_identifiability,
    resolve_physical_background_state,
    validate_profile_inversion_contract,
    validate_spectral_response_operator,
)


def test_omitted_background_is_not_zero_and_zero_requires_explicit_declaration():
    omitted = resolve_physical_background_state(None)
    assert omitted["physical_background_zero"] is None
    assert omitted["background_state"] == "uncharacterized_constant_offset"
    explicit = resolve_physical_background_state("assumed_zero")
    assert explicit["physical_background_zero"] is True
    with pytest.raises(ValueError):
        resolve_physical_background_state("assumed_zero", measured_background_spectrum=[1.0, 2.0])


def test_reference_background_requires_actual_reference_spectra():
    with pytest.raises(ValueError):
        resolve_physical_background_state("reference_pca")
    result = resolve_physical_background_state("reference_pca", reference_spectra=np.ones((3, 8)))
    assert result["background_identifiability"] == "conditional_on_supplied_reference_basis"


def test_measured_response_is_validated_but_never_renormalized():
    response = np.eye(4)
    assert np.array_equal(validate_spectral_response_operator(response, 4), response)
    bad = response.copy()
    bad[0, 0] = 0.8
    with pytest.raises(ValueError, match="already sum to one"):
        validate_spectral_response_operator(bad, 4)


def test_covariance_components_are_combined_once_and_discrepancy_must_be_psd():
    result = compose_spectral_covariance(
        3, measurement_sigma=np.array([1.0, 2.0, 3.0]),
        model_discrepancy_covariance=0.1 * np.eye(3),
    )
    assert np.allclose(result["total_covariance"], np.diag([1.1, 4.1, 9.1]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        compose_spectral_covariance(
            2, measurement_sigma=[1.0, 1.0],
            model_discrepancy_covariance=[[0.0, 2.0], [2.0, 0.0]],
        )


def test_profile_bounds_and_regularization_are_never_implicit():
    with pytest.raises(ValueError):
        validate_profile_inversion_contract(None, 0.1, ["ne"], regularization_scope="curvature")
    with pytest.raises(ValueError):
        validate_profile_inversion_contract((0.1, 2.0), None, ["ne"], regularization_scope="curvature")
    result = validate_profile_inversion_contract(
        (0.1, 2.0), {"ne": 0.02, "Te": 0.04}, ["ne", "Te"],
        regularization_scope="second_difference_of_log_relative_profiles",
    )
    assert result["regularization_strength"] == {"ne": 0.02, "Te": 0.04}


def test_local_identifiability_reports_null_directions_and_zero_sensitivity():
    j = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    result = diagnose_local_identifiability(j, ["ne", "Te", "flow"])
    assert result["jacobian_rank"] == 1
    assert result["status"] == "structurally_or_practically_rank_deficient"
    assert result["zero_sensitivity_parameters"] == ["flow"]
    assert result["unconstrained_parameter_directions_scaled"].shape[0] == 2
