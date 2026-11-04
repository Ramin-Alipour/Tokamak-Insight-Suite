import numpy as np
import pytest

from tokamak_analysis.cts_stokes_transfer import (
    propagate_stokes_slab,
    stokes_mueller_absorption_matrix,
)


def test_unpolarized_scalar_limit_matches_formal_radiative_transfer_solution():
    alpha = 0.37
    emissivity = 2.4
    length = 0.8
    incident = np.array([1.1, 0.0, 0.0, 0.0])
    matrix = stokes_mueller_absorption_matrix(alpha, 0, 0, 0, 0, 0, 0)
    result = propagate_stokes_slab(
        incident, np.array([emissivity, 0, 0, 0]), matrix, length
    )
    expected = incident[0] * np.exp(-alpha * length) + (
        emissivity / alpha
    ) * (1.0 - np.exp(-alpha * length))
    assert result["I"] == pytest.approx(expected, rel=2e-13)
    np.testing.assert_allclose(result["stokes_radiance"][1:], 0.0, atol=1e-14)


def test_zero_absorption_faraday_rotation_conserves_intensity_and_polarization():
    matrix = stokes_mueller_absorption_matrix(0, 0, 0, 0, 0, 0, 2.0)
    incident = np.array([3.0, 1.0, 0.0, 0.0])
    result = propagate_stokes_slab(
        incident, np.zeros(4), matrix, np.pi / 8
    )
    assert result["I"] == pytest.approx(3.0, abs=1e-13)
    assert np.linalg.norm(result["stokes_radiance"][1:]) == pytest.approx(1.0, abs=1e-13)
    assert result["Q"] == pytest.approx(np.cos(np.pi / 4), abs=1e-13)
    assert result["U"] == pytest.approx(np.sin(np.pi / 4), abs=1e-13)


def test_zero_matrix_handles_emission_without_matrix_inverse():
    incident = np.array([0.0, 0.0, 0.0, 0.0])
    emissivity = np.array([2.0, 0.2, 0.0, 0.0])
    result = propagate_stokes_slab(incident, emissivity, np.zeros((4, 4)), 0.5)
    np.testing.assert_allclose(result["stokes_radiance"], 0.5 * emissivity)


@pytest.mark.parametrize(
    "coefficients",
    [
        (-0.1, 0, 0, 0, 0, 0, 0),
        (0.1, 0.2, 0, 0, 0, 0, 0),
    ],
)
def test_mueller_builder_rejects_non_passive_absorption(coefficients):
    with pytest.raises(ValueError, match="passive-medium absorption"):
        stokes_mueller_absorption_matrix(*coefficients)


def test_stokes_propagation_rejects_nonphysical_incident_polarization():
    with pytest.raises(ValueError, match="not physically realizable"):
        propagate_stokes_slab(
            [1.0, 1.1, 0.0, 0.0], [0, 0, 0, 0], np.zeros((4, 4)), 1.0
        )


def test_emissivity_must_lie_inside_physical_polarization_cone():
    with pytest.raises(ValueError, match="emissivity Stokes vector"):
        propagate_stokes_slab(
            [1.0, 0.0, 0.0, 0.0], [1.0, 1.1, 0.0, 0.0],
            np.zeros((4, 4)), 1.0
        )
