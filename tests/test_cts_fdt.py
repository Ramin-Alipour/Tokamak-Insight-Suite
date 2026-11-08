import numpy as np
import pytest
from tokamak_analysis.cts_fdt import density_fluctuation_spectral_matrix


def test_scalar_passive_relaxation_obeys_classical_fdt_and_is_positive():
    a, tau, temperature = 3.0e19, 2.0e-6, 100.0 * 1.602176634e-19
    omega = np.array([-2.0e5, 2.0e5])
    chi = (-a / (1.0 - 1j * omega * tau))[:, None, None]
    out = density_fluctuation_spectral_matrix(chi, omega, temperature)
    expected = -2.0 * temperature * chi.imag[:, 0, 0] / omega
    np.testing.assert_allclose(out["density_fluctuation_spectral_matrix_m3_s"][:, 0, 0].real, expected)
    assert np.all(out["density_fluctuation_spectral_matrix_m3_s"][:, 0, 0].real > 0)


def test_matrix_fdt_rejects_non_passive_response():
    chi = np.array([[[1.0j, 0.0], [0.0, 1.0j]]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        density_fluctuation_spectral_matrix(chi, [1.0], 1.0)


def test_zero_frequency_requires_explicit_slope_and_uses_continuous_limit():
    chi = np.zeros((1, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match="zero-frequency"):
        density_fluctuation_spectral_matrix(chi, [0.0], 2.0)
    out = density_fluctuation_spectral_matrix(
        chi, [0.0], 2.0, zero_frequency_imaginary_slope=np.array([[-0.25]])
    )
    assert out["density_fluctuation_spectral_matrix_m3_s"][0, 0, 0].real == pytest.approx(1.0)


def test_fdt_rejects_shape_mismatch_and_nonpositive_temperature():
    with pytest.raises(ValueError, match="match the response"):
        density_fluctuation_spectral_matrix(np.zeros((2, 1, 1)), [1.0], 1.0)
    with pytest.raises(ValueError, match="temperature"):
        density_fluctuation_spectral_matrix(np.zeros((1, 1, 1)), [1.0], 0.0)
