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



def test_nonrelativistic_collective_tabulated_evdf_rejects_superluminal_support():
    import pytest
    from tokamak_analysis.thomson_generalized_dispersion import (
        collective_tabulated_electron_spectral_density,
    )

    wavelength = np.linspace(1060.0, 1068.0, 41)
    velocity = np.linspace(-1.01 * 299792458.0, 1.01 * 299792458.0, 101)
    with pytest.raises(ValueError, match="non-relativistic tabulated-Landau CTS kernel"):
        collective_tabulated_electron_spectral_density(
            wavelength, 1064.0, np.pi / 2.0,
            electron_density_m3=1.0e18,
            electron_velocity_grid_m_s=velocity,
            electron_probability_density_s_m=np.exp(-0.5 * (velocity / 1.0e8) ** 2),
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0], ion_temperature_eV=[10.0],
            electron_velocity_m_s=[0.0, 0.0, 0.0],
            ion_velocity_m_s=[[0.0, 0.0, 0.0]],
            incident_direction=[1.0, 0.0, 0.0],
            scattered_direction=[0.0, 1.0, 0.0],
        )


def test_nonrelativistic_collective_tabulated_evdf_rejects_superluminal_ion_support():
    import pytest
    from tokamak_analysis.thomson_generalized_dispersion import (
        collective_tabulated_electron_spectral_density,
    )

    wavelength = np.linspace(1060.0, 1068.0, 41)
    electron_velocity = np.linspace(-8.0e6, 8.0e6, 101)
    ion_velocity = np.linspace(-1.01 * 299792458.0, 1.01 * 299792458.0, 101)
    with pytest.raises(ValueError, match="projected ion velocity support"):
        collective_tabulated_electron_spectral_density(
            wavelength, 1064.0, np.pi / 2.0,
            electron_density_m3=1.0e18,
            electron_velocity_grid_m_s=electron_velocity,
            electron_probability_density_s_m=np.exp(-0.5 * (electron_velocity / 1.0e6) ** 2),
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0],
            electron_velocity_m_s=[0.0, 0.0, 0.0],
            ion_velocity_m_s=[[0.0, 0.0, 0.0]],
            incident_direction=[1.0, 0.0, 0.0],
            scattered_direction=[0.0, 1.0, 0.0],
            ion_velocity_grids_m_s=ion_velocity[None, :],
            ion_probability_densities_s_m=np.exp(-0.5 * (ion_velocity / 1.0e8) ** 2)[None, :],
        )


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
    from tokamak_analysis.thomson_generalized_dispersion import collective_tabulated_electron_spectral_density
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



def test_tabulated_evdf_phase_derivative_formula_registry_is_synchronized():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root/"FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root/"tis"/"data"/"FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "cts_tabulated_evdf_landau_phase_derivative"
    matches = [item for item in formula if item.get("execution_contract", {}).get("formula_key") == key]
    assert len(matches) == 1
    assert formula == packaged
    assert matches[0]["execution_contract"]["function_link"] == "tokamak_analysis.thomson_generalized_dispersion.generalized_electron_susceptibility"


def test_exact_sokhotski_plemelj_landau_limit_matches_principal_value_and_resonance():
    velocity = np.linspace(-12.0, 12.0, 2401)
    distribution = np.exp(-0.5*(velocity/2.0)**2)
    phase_velocity = 0.735
    result = generalized_landau_integral(velocity, distribution, phase_velocity, landau_limit=True)
    normalized = result["normalized_probability_density_s_m"]
    slopes = np.diff(normalized)/np.diff(velocity)
    interval = np.searchsorted(velocity, phase_velocity)-1
    assert result["pole_imaginary_part_m_s"] == 0.0
    assert np.isclose(result["integral"].imag, np.pi*slopes[interval], rtol=1e-12, atol=1e-14)
    assert result["landau_limit"] is True
    assert "sokhotski_plemelj" in result["quadrature"]


def test_exact_landau_limit_rejects_derivative_knots_and_higher_order_poles():
    import pytest
    velocity = np.linspace(-5.0, 5.0, 101)
    distribution = np.exp(-0.5*velocity**2)
    with pytest.raises(ValueError, match="derivative knot"):
        generalized_landau_integral(velocity, distribution, 0.0, landau_limit=True)
    with pytest.raises(ValueError, match="first-order"):
        generalized_landau_integral(velocity, distribution, 0.37, landau_limit=True, pole_order=2)


def test_exact_landau_limit_is_connected_to_electron_susceptibility():
    velocity = np.linspace(-12.0, 12.0, 2401)
    distribution = np.exp(-0.5*(velocity/2.0)**2)
    result = generalized_electron_susceptibility(
        velocity, distribution, 0.73, electron_density_m3=2.0e18,
        wavenumber_m_inverse=120.0, landau_limit=True,
    )
    assert result["landau_limit"] is True
    assert np.isfinite(result["electron_susceptibility"].real)
    assert np.isfinite(result["electron_susceptibility"].imag)
    assert result["electron_susceptibility"].imag != 0.0


def test_exact_landau_limit_is_exposed_by_public_advanced_cts_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    velocity = np.linspace(-12.0, 12.0, 2401)
    distribution = np.exp(-0.5*(velocity/2.0)**2)
    result = run_advanced_thomson_physics(
        "cts_tabulated_evdf_susceptibility", velocity_grid_m_s=velocity,
        probability_density_s_m=distribution, phase_velocity_m_s=0.735,
        electron_density_m3=1.0e18, wavenumber_m_inverse=100.0, landau_limit=True,
    )
    assert result["landau_limit"] is True
    assert np.isfinite(result["electron_susceptibility"].real)
    assert np.isfinite(result["electron_susceptibility"].imag)



def test_collective_tabulated_evdf_accepts_non_maxwellian_tabulated_ion_population():
    from tokamak_analysis.thomson_generalized_dispersion import (
        collective_tabulated_electron_spectral_density,
    )
    wavelength = np.linspace(1060.0, 1068.0, 41)
    velocity = np.linspace(-8.0e6, 8.0e6, 1601)
    electron_pdf = np.exp(-0.5 * (velocity / 1.0e6) ** 2)
    ion_velocity = np.linspace(-2.0e5, 2.0e5, 1201)
    ion_pdf = (
        np.exp(-0.5*((ion_velocity-2.5e4)/2.0e4)**2)
        + 0.6*np.exp(-0.5*((ion_velocity+3.0e4)/1.5e4)**2)
    )
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    result = run_advanced_thomson_physics(
        "cts_tabulated_evdf_landau",
        wavelength, 1064.0, np.pi/2,
        electron_density_m3=1.0e18,
        electron_velocity_grid_m_s=velocity,
        electron_probability_density_s_m=electron_pdf,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        ion_velocity_grids_m_s=ion_velocity[None, :],
        ion_probability_densities_s_m=ion_pdf[None, :],
    )
    assert result["ion_distribution_model"] == "researcher_supplied_tabulated_projected_1D_EVDF_per_species"
    assert result["ion_susceptibility"].shape == (1, wavelength.size)
    assert np.all(np.isfinite(result["ion_susceptibility"]))
    assert np.all(np.isfinite(result["spectral_density_per_nm"]))
    assert np.all(result["spectral_density_per_nm"] >= 0.0)


def test_collective_tabulated_evdf_requires_ion_grid_and_density_together():
    from tokamak_analysis.thomson_generalized_dispersion import (
        collective_tabulated_electron_spectral_density,
    )
    wavelength = np.linspace(1060.0, 1068.0, 41)
    velocity = np.linspace(-8.0e6, 8.0e6, 1601)
    with np.testing.assert_raises_regex(ValueError, "must be supplied together"):
        collective_tabulated_electron_spectral_density(
            wavelength, 1064.0, np.pi/2,
            electron_density_m3=1.0e18,
            electron_velocity_grid_m_s=velocity,
            electron_probability_density_s_m=np.exp(-0.5*(velocity/1e6)**2),
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}], ion_fractions=[1.0],
            electron_velocity_m_s=[0.0, 0.0, 0.0],
            ion_velocity_m_s=[[0.0, 0.0, 0.0]],
            incident_direction=[1.0, 0.0, 0.0],
            scattered_direction=[0.0, 1.0, 0.0],
            ion_velocity_grids_m_s=velocity[None, :],
        )


def test_tabulated_cts_evdfs_return_directional_temperature_and_shape_moments_through_public_api():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    wavelength = np.linspace(1060.0, 1068.0, 81)
    te, ti = 120.0, 18.0
    me, mi, charge = 9.1093837015e-31, 2.0 * 1.66053906660e-27, 1.602176634e-19
    se, si = np.sqrt(te * charge / me), np.sqrt(ti * charge / mi)
    ve = np.linspace(-8.0 * se, 8.0 * se, 2401)
    vi = np.linspace(-8.0 * si, 8.0 * si, 2401)
    fe, fi = np.exp(-0.5 * (ve / se) ** 2), np.exp(-0.5 * (vi / si) ** 2)
    result = run_advanced_thomson_physics(
        "cts_tabulated_evdf_landau", wavelength, 1064.0, np.pi / 2.0,
        electron_density_m3=1.0e18, electron_velocity_grid_m_s=ve,
        electron_probability_density_s_m=fe,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.0}],
        ion_fractions=[1.0], electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]], incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        ion_velocity_grids_m_s=vi[None, :], ion_probability_densities_s_m=fi[None, :],
    )
    em, im = result["electron_projected_moments"], result["ion_projected_moments"][0]
    assert np.isclose(em["directional_second_moment_temperature_eV"], te, rtol=2e-4)
    assert np.isclose(im["directional_second_moment_temperature_eV"], ti, rtol=2e-4)
    assert abs(em["standardized_skewness"]) < 1e-10
    assert abs(im["standardized_skewness"]) < 1e-10
    assert abs(em["excess_kurtosis"]) < 1e-3
    assert abs(im["excess_kurtosis"]) < 1e-3
    assert im["species"] == "D+"


def test_maxwellian_ion_moment_output_matches_analytic_temperature():
    from tokamak_analysis.thomson_generalized_dispersion import collective_tabulated_electron_spectral_density

    wavelength = np.linspace(1060.0, 1068.0, 41)
    velocity = np.linspace(-8.0e6, 8.0e6, 1601)
    result = collective_tabulated_electron_spectral_density(
        wavelength, 1064.0, np.pi / 2.0, electron_density_m3=1.0e18,
        electron_velocity_grid_m_s=velocity,
        electron_probability_density_s_m=np.exp(-0.5 * (velocity / 1.0e6) ** 2),
        ion_species=[{"name": "H+", "charge_state": 1.0, "mass_amu": 1.00784}],
        ion_fractions=[1.0], ion_temperature_eV=[12.5],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[1.0, 0.0, 0.0], scattered_direction=[0.0, 1.0, 0.0],
        pole_imaginary_part_m_s=100.0,
    )
    assert result["ion_projected_moments"][0]["directional_second_moment_temperature_eV"] == 12.5
    assert result["ion_projected_moments"][0]["standardized_skewness"] == 0.0


def test_projected_evdf_moment_formula_registry_is_synchronized_and_executable():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert formula == packaged
    matches = [
        item for item in formula
        if item.get("execution_contract", {}).get("formula_key") == "cts_tabulated_evdf_projected_moments"
    ]
    assert len(matches) == 1
    contract = matches[0]["execution_contract"]
    assert contract["implementation_link_status"] == "mapped"
    assert contract["function_link"] == "tokamak_analysis.thomson_generalized_dispersion.collective_tabulated_electron_spectral_density"
    assert "tests/test_thomson_generalized_dispersion.py::test_tabulated_cts_evdfs_return_directional_temperature_and_shape_moments_through_public_api" in contract["test_links"]
