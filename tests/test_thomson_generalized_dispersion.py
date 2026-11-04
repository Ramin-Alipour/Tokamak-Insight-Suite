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
    assert "one complex pole per call" in result["limitation"]


def test_collective_tabulated_evdf_forward_model_returns_finite_positive_spectrum():
    from tokamak_analysis.thomson_generalized_dispersion import (
        collective_tabulated_electron_spectral_density,
    )
    wavelength = np.linspace(1060.0, 1068.0, 41)
    velocity = np.linspace(-8.0e6, 8.0e6, 1601)
    sigma_e = 1.0e6
    distribution = np.exp(-0.5 * (velocity / sigma_e) ** 2)
    result = collective_tabulated_electron_spectral_density(
        wavelength, 1064.0, np.pi / 2.0,
        electron_density_m3=1.0e18,
        electron_velocity_grid_m_s=velocity,
        electron_probability_density_s_m=distribution,
        ion_species=[{"name": "H+", "charge_state": 1.0, "mass_amu": 1.00784}],
        ion_fractions=[1.0],
        ion_temperature_eV=[10.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        pole_imaginary_part_m_s=100.0,
    )
    spectrum = result["spectral_density_per_nm"]
    assert spectrum.shape == wavelength.shape
    assert np.all(np.isfinite(spectrum))
    assert np.any(spectrum > 0)
    assert result["electron_distribution_model"] == "researcher_supplied_tabulated_projected_1D_EVDF"


def test_generalized_landau_integral_supports_higher_order_pole():
    velocity = np.linspace(-10.0, 10.0, 1001)
    distribution = np.exp(-0.5 * velocity**2)
    result = generalized_landau_integral(
        velocity, distribution, 0.35,
        pole_imaginary_part_m_s=0.05,
        pole_order=2,
    )
    assert result["pole_order"] == 2
    assert np.isfinite(result["integral"].real)
    assert np.isfinite(result["integral"].imag)


def test_generalized_landau_integral_includes_distributional_finite_support_edges():
    # A constant top-hat on [a,b], extended by zero outside, has no interior
    # derivative. Its complete distributional derivative consists only of
    # +f(a) delta(v-a) - f(b) delta(v-b).
    velocity = np.linspace(-2.0, 3.0, 101)
    distribution = np.ones_like(velocity)
    phase_velocity = 0.4
    eta = 0.07
    result = generalized_landau_integral(
        velocity, distribution, phase_velocity,
        pole_imaginary_part_m_s=eta,
    )
    normalized = result["normalized_probability_density_s_m"]
    zeta = complex(phase_velocity, eta)
    expected = normalized[0] / (velocity[0] - zeta) - normalized[-1] / (velocity[-1] - zeta)
    assert result["interior_derivative_integral"] == 0.0j
    assert result["finite_support_boundary_integral"] == expected
    assert result["integral"] == expected
    assert result["finite_support_extension"] == "zero_outside_tabulated_velocity_interval"


def test_collective_tabulated_evdf_accepts_compact_support_and_includes_edge_terms():
    wavelength = np.linspace(1060.0, 1068.0, 41)
    velocity = np.linspace(-8.0e6, 8.0e6, 1601)
    # A compact-support top-hat has finite non-zero endpoint values. Under
    # the model's declared zero extension, its derivative includes both
    # endpoint delta terms, which generalized_landau_integral now evaluates.
    distribution = np.ones_like(velocity)
    result = collective_tabulated_electron_spectral_density(
        wavelength, 1064.0, np.pi / 2.0,
        electron_density_m3=1.0e18,
        electron_velocity_grid_m_s=velocity,
        electron_probability_density_s_m=distribution,
        ion_species=[{"name": "H+", "charge_state": 1.0, "mass_amu": 1.00784}],
        ion_fractions=[1.0],
        ion_temperature_eV=[10.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        pole_imaginary_part_m_s=100.0,
    )
    assert np.all(np.isfinite(result["spectral_density_per_nm"]))
    assert np.any(result["spectral_density_per_nm"] > 0.0)
    assert "distributional_finite_support_edges" in result["pole_quadrature"]
    assert result["velocity_support_extension"].startswith("zero_outside_supplied_interval")
