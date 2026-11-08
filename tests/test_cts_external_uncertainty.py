import numpy as np
import pytest

from tokamak_analysis.cts_external_uncertainty import (
    validate_external_input_covariance,
    sample_external_input_vectors,
)


def test_local_full_em_covariance_contract_and_order_are_preserved():
    order = ("magnetic_field_T_x", "magnetic_field_T_y")
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    returned_order, returned = validate_external_input_covariance(covariance, order)
    assert returned_order == order
    np.testing.assert_allclose(returned, covariance)


def test_covariance_accepts_singular_positive_semidefinite_matrix():
    _, returned = validate_external_input_covariance(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        ("magnetic_field_T_x", "magnetic_field_T_y"),
    )
    assert np.linalg.eigvalsh(returned).min() >= -1e-12


@pytest.mark.parametrize("matrix,order,match", [
    (np.eye(2), ("magnetic_field_T_x",), "match"),
    (np.array([[1.0, 2.0], [0.0, 1.0]]),
     ("magnetic_field_T_x", "magnetic_field_T_y"), "symmetric"),
    (np.array([[1.0, 2.0], [2.0, 1.0]]),
     ("magnetic_field_T_x", "magnetic_field_T_y"), "positive semidefinite"),
    (np.eye(1), ("not_supported",), "unsupported"),
    (np.eye(2), ("magnetic_field_T_x", "magnetic_field_T_x"), "unique"),
])
def test_invalid_covariance_contracts_are_rejected(matrix, order, match):
    with pytest.raises(ValueError, match=match):
        validate_external_input_covariance(matrix, order)


def test_local_external_covariance_sampling_is_reproducible():
    base = np.array([2.0, -1.0])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    first = sample_external_input_vectors(base, covariance, samples=24, seed=17)
    second = sample_external_input_vectors(base, covariance, samples=24, seed=17)
    assert first.shape == (24, 2)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("covariance,match", [
    (np.array([[1.0, 2.0], [0.0, 1.0]]), "symmetric"),
    (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
    (np.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]), "real numeric"),
])
def test_sampler_rejects_invalid_covariance(covariance, match):
    with pytest.raises(ValueError, match=match):
        sample_external_input_vectors(np.array([0.0, 0.0]), covariance, samples=24)


def test_sampler_rejects_non_integer_seed():
    with pytest.raises(ValueError, match="seed must be an integer"):
        sample_external_input_vectors(
            np.array([0.0, 0.0]), np.eye(2), samples=24, seed=1.5
        )


def test_shared_external_jacobian_propagation_retains_cross_covariance():
    from tokamak_analysis.thomson_external_uncertainty import propagate_external_covariance

    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = propagate_external_covariance(
        lambda x: np.array([x[0] + 2.0*x[1]]), [2.0, 3.0], covariance
    )
    jacobian = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(
        result["external_output_covariance"], jacobian @ covariance @ jacobian.T,
        rtol=2e-6,
    )
    assert "distinct_from_measurement_noise" in result["covariance_semantics"]


def test_shared_external_monte_carlo_is_reproducible():
    from tokamak_analysis.thomson_external_uncertainty import monte_carlo_external_covariance

    covariance = np.array([[1.0, 0.7], [0.7, 1.0]])
    function = lambda x: np.array([x[0] + x[1]])
    first = monte_carlo_external_covariance(function, [0.0, 0.0], covariance, samples=1000, seed=31)
    second = monte_carlo_external_covariance(function, [0.0, 0.0], covariance, samples=1000, seed=31)
    np.testing.assert_array_equal(first["input_samples"], second["input_samples"])
    assert first["external_output_covariance"][0, 0] == pytest.approx(3.4, rel=0.12)


def test_cts_api_propagates_correlated_external_inputs_through_inversion():
    from tokamak_analysis.cts_external_uncertainty import propagate_cts_external_uncertainty

    names = ("magnetic_field_T_x", "magnetic_field_T_y")
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = propagate_cts_external_uncertainty(
        lambda x: np.array([x[0] + x[1]]), [2.0, 3.0], covariance,
        input_names=names,
    )
    assert result["input_names"] == names
    assert result["external_output_covariance"][0, 0] == pytest.approx(0.15, rel=2e-6)
