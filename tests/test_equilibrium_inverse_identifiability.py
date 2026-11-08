"""Regression tests for inverse-equilibrium identifiability checks."""
import numpy as np
import pytest

from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_inverse


def test_regularization_does_not_hide_rank_deficient_measurement_response():
    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    rr, zz = np.meshgrid(R, Z)
    basis = np.stack((rr + zz, 2.0 * rr - zz))
    # Every sensor observes only the same linear combination of both coefficients.
    response = np.array([[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0]])
    measured = np.array([3.0, 6.0, -3.0])
    sigma = np.ones(3)

    with pytest.raises(ValueError, match=r"not identifiable.*response rank 1 < 2"):
        reconstruct_equilibrium_inverse(
            R, Z, basis,
            magnetic_probe_response_T_per_coefficient=response,
            magnetic_probe_values_T=measured,
            magnetic_probe_sigma_T=sigma,
            regularization=1e-3,
        )
