import numpy as np

from tokamak_analysis.thomson_generalized_dispersion import (
    generalized_landau_integral,
    generalized_electron_susceptibility,
)


def test_generalized_landau_integral_normalizes_tabulated_vdf_and_is_finite():
    velocity = np.linspace(-12.0, 12.0, 1201)
    sigma = 2.0
    distribution = np.exp(-0.5 * (velocity / sigma) ** 2)
    result = generalized_landau_integral(
        velocity, distribution, 0.7,
        pole_imaginary_part_m_s=0.02,
    )
    normalized = result["normalized_probability_density_s_m"]
    assert np.isclose(np.trapz(normalized, velocity), 1.0, rtol=1e-8)
    assert np.isfinite(result["integral"].real)
    assert np.isfinite(result["integral"].imag)
    assert result["pole_order"] == 1


def test_generalized_electron_susceptibility_has_explicit_retarded_pole():
    velocity = np.linspace(-15.0, 15.0, 1501)
    distribution = np.exp(-0.5 * (velocity / 3.0) ** 2)
    result = generalized_electron_susceptibility(
        velocity, distribution, 1.0,
        electron_density_m3=2.0e18,
        wavenumber_m_inverse=120.0,
        pole_imaginary_part_m_s=0.03,
    )
    assert np.isfinite(result["electron_susceptibility"].real)
    assert np.isfinite(result["electron_susceptibility"].imag)
    assert result["pole"].imag > 0
    assert "first-order Landau pole only" in result["limitation"]
