import numpy as np
import pytest
from scipy.special import wofz

from tokamak_analysis.cts_kappa_dispersion import kappa_dispersion_function


def test_kappa_dispersion_has_correct_maxwellian_limit():
    zeta = 0.7 + 0.35j
    result = kappa_dispersion_function(zeta.real, zeta.imag, 1.0e5)
    plasma_z = 1j*np.sqrt(np.pi)*wofz(zeta)
    expected = -2.0*(1.0 + zeta*plasma_z)
    np.testing.assert_allclose(result["kappa_dispersion_integral"], expected,
                               rtol=2.0e-4, atol=2.0e-5)
    assert result["tail_treatment"].startswith("analytic_distribution")


def test_kappa_dispersion_derivative_matches_complex_step_difference():
    zeta = 0.45 + 0.6j
    result = kappa_dispersion_function(zeta.real, zeta.imag, 4.0)
    h = 2.0e-5
    plus = kappa_dispersion_function((zeta+h).real, zeta.imag, 4.0)
    minus = kappa_dispersion_function((zeta-h).real, zeta.imag, 4.0)
    finite_difference = (
        plus["kappa_dispersion_integral"] - minus["kappa_dispersion_integral"]
    )/(2.0*h)
    np.testing.assert_allclose(
        result["kappa_dispersion_integral_derivative"], finite_difference,
        rtol=3.0e-5, atol=2.0e-7,
    )


def test_real_axis_uses_exact_plemelj_resonance_and_rejects_advanced_branch():
    result = kappa_dispersion_function(0.8, 0.0, 3.5)
    expected_imag = np.pi*(-2.0*0.8/(1.0+0.8**2/3.5))*result["normalization"]*(
        1.0+0.8**2/3.5
    )**(-3.5)
    assert result["kappa_dispersion_integral"].imag == pytest.approx(expected_imag)
    with pytest.raises(ValueError, match="retarded"):
        kappa_dispersion_function(0.0, -0.1, 3.5)
    with pytest.raises(ValueError, match="3/2"):
        kappa_dispersion_function(0.0, 0.1, 1.5)
