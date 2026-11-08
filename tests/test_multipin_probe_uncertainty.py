from __future__ import annotations

import numpy as np
import pytest

from tokamak_analysis.multipin_probe import analyze_multipin_probe


def test_multipin_voltage_covariance_propagates_through_nonuniform_gradient_operator():
    time = np.array([0.0, 1.0, 2.0])
    positions = np.array([0.0, 0.01, 0.03, 0.06])
    potentials = np.array([
        [1.0, 0.9, 0.7, 0.4],
        [1.1, 1.0, 0.8, 0.5],
        [1.2, 1.1, 0.9, 0.6],
    ])
    factor = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.2, 1.0, 0.0, 0.0],
        [0.0, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.3, 1.0],
    ])
    covariance = 1.0e-4 * (factor @ factor.T)

    result = analyze_multipin_probe(
        time, potentials, positions, potential_covariance_V2=covariance
    )
    derivative = result["derivative_operator_per_m"]
    expected = derivative @ covariance @ derivative.T
    np.testing.assert_allclose(
        result["electric_field_covariance_V2_m2"],
        np.broadcast_to(expected, (len(time), 4, 4)),
    )
    np.testing.assert_allclose(
        result["electric_field_standard_uncertainty_V_m"],
        np.sqrt(np.diag(expected))[None, :].repeat(len(time), axis=0),
    )
    assert result["uncertainty_scope"] == "shared_pin_covariance"
    assert np.all(np.linalg.eigvalsh(result["electric_field_covariance_V2_m2"]) >= -1e-12)


def test_multipin_accepts_time_resolved_voltage_covariance():
    time = np.array([0.0, 1.0])
    positions = np.array([0.0, 0.5, 1.0])
    potentials = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    covariance = np.stack([np.eye(3) * 0.01, np.eye(3) * 0.04])

    result = analyze_multipin_probe(
        time, potentials, positions, potential_covariance_V2=covariance
    )
    assert result["uncertainty_scope"] == "time_resolved_pin_covariance"
    assert result["electric_field_standard_uncertainty_V_m"].shape == (2, 3)
    assert np.all(
        result["electric_field_standard_uncertainty_V_m"][1]
        > result["electric_field_standard_uncertainty_V_m"][0]
    )


@pytest.mark.parametrize(
    "covariance, message",
    [
        (np.array([[1.0, 0.2], [0.0, 1.0]]), "must be symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
        (np.eye(4), "must have shape"),
    ],
)
def test_multipin_rejects_invalid_voltage_covariance(covariance, message):
    with pytest.raises(ValueError, match=message):
        analyze_multipin_probe(
            [0.0, 1.0], [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]],
            [0.0, 0.5, 1.0], potential_covariance_V2=covariance,
        )
