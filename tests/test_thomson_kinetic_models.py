import numpy as np
import pytest

from tokamak_analysis.thomson_kinetic_models import (
    relativistic_maxwell_juttner_spectrum,
    collective_kappa_spectral_density,
)
from tokamak_analysis.thomson_gap_closure import (
    reconstruct_relativistic_projected_thomson_evdf,
)


def test_relativistic_maxwell_juttner_spectrum_is_normalized_and_finite():
    wavelength = np.linspace(800.0, 1328.0, 801)
    result = relativistic_maxwell_juttner_spectrum(
        wavelength, 1064.0, np.pi/2.0, 500.0,
        flow_projection_m_s=1.0e6, incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
    )
    spectrum = result["spectral_density_per_nm"]
    assert spectrum.shape == wavelength.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0)
    np.testing.assert_allclose(np.trapezoid(spectrum, wavelength), 1.0, rtol=2e-3)
    assert result["captured_probability"] >= 0.995
    assert result["relativistic_kinematics"] is True
    assert result["recoil_included"] is False


def test_collective_kappa_uses_dielectric_response_and_non_maxwellian_electrons():
    wavelength = np.linspace(1040.0, 1088.0, 241)
    common = dict(
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2.0,
        electron_density_m3=1.0e19,
        electron_temperature_eV=150.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[-np.sin(np.pi/4), np.cos(np.pi/4), 0.0],
        scattered_direction=[np.sin(np.pi/4), np.cos(np.pi/4), 0.0],
    )
    low_kappa = collective_kappa_spectral_density(
        wavelength, kappa=2.2, **common
    )
    high_kappa = collective_kappa_spectral_density(
        wavelength, kappa=8.0, **common
    )
    assert low_kappa["spectral_density_per_nm"].shape == wavelength.shape
    assert np.all(np.isfinite(low_kappa["dielectric_function"]))
    assert np.all(np.isfinite(low_kappa["electron_susceptibility"]))
    assert np.all(low_kappa["spectral_density_per_nm"] >= 0)
    assert not np.allclose(
        low_kappa["spectral_density_per_nm"],
        high_kappa["spectral_density_per_nm"],
        rtol=1e-3, atol=0.0,
    )
    assert low_kappa["model"].startswith("unmagnetized collective Thomson")


@pytest.mark.parametrize("include_compton_recoil", [False, True])
def test_relativistic_maxwell_juttner_spectrum_fit_recovers_synthetic_parameters(include_compton_recoil):
    from tokamak_analysis.thomson_kinetic_models import fit_relativistic_maxwell_juttner

    wavelength = np.linspace(800.0, 1328.0, 401)
    angle = np.pi/2.0
    laser = 1064.0
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    scattered = np.array([np.sin(angle/2), np.cos(angle/2), 0.0])
    forward = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 500.0,
        flow_projection_m_s=1.0e6, incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
        include_compton_recoil=include_compton_recoil,
    )
    signal = 2.0e4*forward["spectral_density_per_nm"] + 10.0 + 0.001*(wavelength-laser)
    result = fit_relativistic_maxwell_juttner(
        wavelength, signal, laser, angle, np.eye(wavelength.size),
        np.full(wavelength.size, 1.0),
        incident_direction=incident, scattered_direction=scattered,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
        include_compton_recoil=include_compton_recoil,
    )
    assert result["electron_temperature_eV"] == pytest.approx(500.0, rel=0.15)
    assert result["electron_velocity_projection_m_s"] == pytest.approx(1.0e6, abs=2.0e5)
    assert result["relativistic_kinematics"] is True
    assert result["recoil_included"] is include_compton_recoil
    assert ("Klein-Nishina" in " ".join(result["assumptions"])) is include_compton_recoil
    assert result["momentum_probability_coverage"] >= 0.999
    assert result["momentum_tail_renormalization_performed"] is False
    assert result["fit_covariance"].shape == (5, 5)

@pytest.mark.parametrize("use_bounds", [False, True])
def test_collective_kappa_fit_runs_with_and_without_explicit_parameter_bounds(use_bounds):
    from tokamak_analysis.thomson_kinetic_models import fit_collective_kappa

    wavelength = np.linspace(1048.0, 1080.0, 101)
    angle = np.pi / 2.0
    laser = 1064.0
    incident = np.array([-np.sin(angle / 2.0), np.cos(angle / 2.0), 0.0])
    scattered = np.array([np.sin(angle / 2.0), np.cos(angle / 2.0), 0.0])
    physical_inputs = dict(
        electron_density_m3=1.0e19,
        electron_temperature_eV=150.0,
        kappa=3.5,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident,
        scattered_direction=scattered,
    )
    spectrum = collective_kappa_spectral_density(
        wavelength, laser, angle, **physical_inputs
    )["spectral_density_per_nm"]
    signal = 2.0e4 * spectrum + 10.0 + 0.001 * (wavelength - laser)
    fit_inputs = dict(
        wavelength_nm=wavelength,
        signal=signal,
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        response=np.eye(wavelength.size),
        uncertainty=np.full(wavelength.size, 0.01 * np.max(signal)),
        incident_direction=incident,
        scattered_direction=scattered,
        ion_species=physical_inputs["ion_species"],
        ion_fractions=physical_inputs["ion_fractions"],
        ion_temperature_eV=physical_inputs["ion_temperature_eV"],
        electron_velocity_m_s=physical_inputs["electron_velocity_m_s"],
        ion_velocity_m_s=physical_inputs["ion_velocity_m_s"],
        initial_density_m3=1.0e19,
        initial_temperature_eV=150.0,
        initial_kappa=3.5,
    )
    if use_bounds:
        fit_inputs.update(
            density_bounds_m3=[5.0e18, 2.0e19],
            temperature_bounds_eV=[75.0, 300.0],
            kappa_bounds=[2.0, 6.0],
        )

    result = fit_collective_kappa(**fit_inputs)

    assert result["optimizer_success"] is True
    assert result["electron_density_m3"] == pytest.approx(1.0e19, rel=2e-3)
    assert result["electron_temperature_eV"] == pytest.approx(150.0, rel=2e-3)
    assert result["kappa_index"] == pytest.approx(3.5, rel=2e-3)
    assert result["fit_covariance"].shape == (3, 3)


def test_relativistic_effective_cross_section_matches_polarized_thomson_limit():
    wavelength = np.linspace(532.0-40.0, 532.0+40.0, 801)
    result = relativistic_maxwell_juttner_spectrum(
        wavelength, 532.0, np.pi/2, 100.0,
        flow_projection_m_s=0.0,
        incident_polarization=np.array([0.0, 0.0, 1.0]),
    )
    classical_radius = 2.8179403262e-15
    assert result["effective_differential_cross_section_m2_sr"] == pytest.approx(
        classical_radius**2, rel=2e-3
    )


def test_relativistic_response_convolution_conserves_area_on_nonuniform_bins():
    from tokamak_analysis.thomson_kinetic_models import _apply_response

    wavelength = np.array([500.0, 501.0, 503.0, 506.0, 510.0, 515.0, 521.0])
    edges = np.empty(wavelength.size+1)
    edges[1:-1] = 0.5*(wavelength[:-1]+wavelength[1:])
    edges[0] = wavelength[0]-0.5*(wavelength[1]-wavelength[0])
    edges[-1] = wavelength[-1]+0.5*(wavelength[-1]-wavelength[-2])
    widths = np.diff(edges)
    signal = np.array([0.2, 0.8, 1.5, 2.0, 1.2, 0.5, 0.1])
    response = np.eye(wavelength.size)*0.8
    response += np.roll(np.eye(wavelength.size), 1, axis=0)*0.2
    result = _apply_response(response, signal, wavelength)
    assert np.sum(result*widths) == pytest.approx(np.sum(signal*widths), rel=1e-12)


def test_relativistic_compton_model_uses_recoil_and_klein_nishina_weighting():
    wavelength = np.linspace(532.0-40.0, 532.0+40.0, 801)
    thomson = relativistic_maxwell_juttner_spectrum(
        wavelength, 532.0, np.pi/2.0, 100.0,
        flow_projection_m_s=0.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
    )
    compton = relativistic_maxwell_juttner_spectrum(
        wavelength, 532.0, np.pi/2.0, 100.0,
        flow_projection_m_s=0.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
        include_compton_recoil=True,
    )
    assert compton["recoil_included"] is True
    assert "Klein_Nishina" in compton["scattering_cross_section_model"]
    assert compton["maximum_recoil_correction_fraction_on_quadrature_support"] > 0.0
    assert compton["effective_differential_cross_section_m2_sr"] > 0.0
    # Recoil transfers energy to the electron and therefore red-shifts the
    # scattered photons relative to the elastic Thomson calculation.
    assert compton["fitted_centroid_nm"] > thomson["fitted_centroid_nm"]
    assert np.all(np.isfinite(compton["spectral_density_per_nm"]))
    np.testing.assert_allclose(
        np.sum(compton["spectral_density_per_nm"]*compton["wavelength_bin_widths_nm"]),
        1.0, rtol=1e-12,
    )


def test_relativistic_compton_mode_rejects_non_boolean_selection():
    wavelength = np.linspace(500.0, 564.0, 101)
    with pytest.raises(ValueError, match="include_compton_recoil must be boolean"):
        relativistic_maxwell_juttner_spectrum(
            wavelength, 532.0, np.pi/2.0, 100.0,
            incident_polarization=[0.0, 0.0, 1.0],
            include_compton_recoil="yes",
        )


def test_landau_susceptibility_matches_exact_maxwellian_plasma_dispersion_function():
    from scipy.special import wofz
    from tokamak_analysis.thomson_kinetic_models import _landau_susceptibility

    sigma = 2.3e6
    phase_velocity = 0.7*sigma
    omega_p2 = 3.0e20
    wave_number = 85.0
    zeta = phase_velocity/(np.sqrt(2.0)*sigma)
    plasma_dispersion = 1j*np.sqrt(np.pi)*wofz(zeta)
    exact_landau_integral = -(1.0 + zeta*plasma_dispersion)/(sigma*sigma)
    expected = -(omega_p2/(wave_number*wave_number))*exact_landau_integral
    actual = _landau_susceptibility(
        phase_velocity, omega_p2, wave_number, sigma, distribution="maxwellian"
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-14)


def test_kappa_landau_susceptibility_remains_finite_near_finite_variance_boundary():
    from tokamak_analysis.thomson_kinetic_models import _landau_susceptibility

    sigma = 1.7e6
    values = [
        _landau_susceptibility(
            0.8*sigma, 2.0e20, 110.0, sigma,
            distribution="kappa", kappa=kappa,
        )
        for kappa in (1.5001, 1.51, 1.7, 3.0)
    ]
    assert np.all(np.isfinite(np.asarray(values).real))
    assert np.all(np.isfinite(np.asarray(values).imag))
    assert len({round(float(value.real), 8) for value in values}) == len(values)


def test_near_critical_kappa_landau_tail_matches_adaptive_infinite_domain_quadrature():
    from scipy.integrate import quad
    from tokamak_analysis.thomson_kinetic_models import (
        _kappa_pdf_and_derivatives,
        _landau_susceptibility,
    )

    sigma = 1.7e6
    phase_velocity = 0.8*sigma
    kappa = 1.5001
    omega_p2 = 2.0e20
    wave_number = 110.0

    def symmetric_integrand(t):
        offset = sigma*t/(1.0-t)
        jacobian = sigma/(1.0-t)**2
        _, derivative_plus, _ = _kappa_pdf_and_derivatives(
            np.asarray(phase_velocity+offset), sigma, kappa
        )
        _, derivative_minus, _ = _kappa_pdf_and_derivatives(
            np.asarray(phase_velocity-offset), sigma, kappa
        )
        return float(jacobian*(derivative_plus-derivative_minus)/offset)

    principal, _ = quad(
        symmetric_integrand, 0.0, 1.0,
        epsabs=1e-25, epsrel=2e-9, limit=1000,
    )
    _, derivative_at, _ = _kappa_pdf_and_derivatives(
        np.asarray(phase_velocity), sigma, kappa
    )
    expected = -(omega_p2/wave_number**2)*(
        principal+1j*np.pi*float(derivative_at)
    )
    actual = _landau_susceptibility(
        phase_velocity, omega_p2, wave_number, sigma,
        distribution="kappa", kappa=kappa,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-14)


def test_gyrotropic_bimaxwellian_multi_pitch_gls_recovers_parallel_and_perpendicular_temperatures():
    from tokamak_analysis.thomson_kinetic_models import infer_gyrotropic_bi_maxwellian_temperatures

    pitch = np.array([1.0, 0.5, 0.0, -0.7])
    t_perp, t_parallel = 80.0, 180.0
    projected = t_perp*(1.0-pitch**2)+t_parallel*pitch**2
    covariance = np.diag([4.0, 9.0, 16.0, 25.0])*1e-2
    result = infer_gyrotropic_bi_maxwellian_temperatures(
        projected, pitch,
        projected_temperature_covariance_eV2=covariance,
    )
    assert result["identifiability"] == "identifiable_under_gyrotropic_bi_maxwellian_assumption"
    assert result["reportable_as_inferred_parameter"] is True
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(t_perp)
    assert result["electron_temperature_parallel_eV"] == pytest.approx(t_parallel)
    assert result["temperature_covariance_eV2"].shape == (2, 2)
    assert result["degrees_of_freedom"] == 2


def test_gyrotropic_bimaxwellian_single_pitch_does_not_invent_anisotropy():
    from tokamak_analysis.thomson_kinetic_models import infer_gyrotropic_bi_maxwellian_temperatures

    result = infer_gyrotropic_bi_maxwellian_temperatures(
        [125.0, 125.0], [0.5, -0.5],
        projected_temperature_covariance_eV2=np.eye(2),
    )
    assert result["identifiability"] == "non_identifiable"
    assert result["reportable_as_inferred_parameter"] is False
    assert result["electron_temperature_parallel_eV"] is None
    assert result["electron_temperature_perpendicular_eV"] is None
    assert result["temperature_covariance_eV2"] is None


def test_gyrotropic_bimaxwellian_gls_rejects_non_positive_definite_covariance():
    from tokamak_analysis.thomson_kinetic_models import infer_gyrotropic_bi_maxwellian_temperatures

    with pytest.raises(ValueError, match="positive definite"):
        infer_gyrotropic_bi_maxwellian_temperatures(
            [180.0, 80.0], [1.0, 0.0],
            projected_temperature_covariance_eV2=np.array([[1.0, 1.0], [1.0, 1.0]]),
        )


def test_thomson_three_dimensional_flow_gls_recovers_vector_from_independent_views():
    from tokamak_analysis.thomson_kinetic_models import infer_three_dimensional_flow_from_thomson_projections

    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2**-0.5, 2**-0.5, 0.0]])
    expected = np.array([1.0e5, -2.0e5, 3.0e5])
    projections = directions @ expected
    result = infer_three_dimensional_flow_from_thomson_projections(
        projections, directions,
        projection_covariance_m2_s2=np.eye(4)*1.0e4,
    )
    assert result["identifiability"] == "identifiable_from_three_dimensional_view_geometry"
    assert result["reportable_as_inferred_parameter"] is True
    np.testing.assert_allclose(result["electron_velocity_vector_m_s"], expected, rtol=1e-10, atol=1e-8)
    assert result["electron_velocity_covariance_m2_s2"].shape == (3, 3)


def test_thomson_three_dimensional_flow_propagates_correlated_direction_covariance():
    from tokamak_analysis.thomson_kinetic_models import infer_three_dimensional_flow_from_thomson_projections

    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2**-0.5, 2**-0.5, 0.0],
    ])
    expected = np.array([1.0e5, -2.0e5, 3.0e5])
    projections = directions @ expected
    result = infer_three_dimensional_flow_from_thomson_projections(
        projections,
        directions,
        projection_covariance_m2_s2=np.eye(4) * 1.0e4,
        scattering_vector_covariance=np.eye(12) * 1.0e-6,
    )

    assert result["geometry_uncertainty_included"] is True
    assert result["geometry_covariance_iterations"] >= 1
    assert result["geometry_covariance_method"] == "iterative_first_order_tangent_plane_errors_in_variables_GLS"
    assert result["identifiability"] == "identifiable_from_three_dimensional_view_geometry"
    assert result["reportable_as_inferred_parameter"] is True
    assert result["tangent_plane_scattering_vector_covariance"].shape == (12, 12)
    assert result["effective_projection_covariance_m2_s2"].shape == (4, 4)
    assert not np.allclose(
        result["effective_projection_covariance_m2_s2"],
        result["input_projection_covariance_m2_s2"],
    )
    np.testing.assert_allclose(result["electron_velocity_vector_m_s"], expected, rtol=1e-5, atol=1e-3)


def test_thomson_three_dimensional_flow_withholds_vector_for_rank_deficient_views():
    from tokamak_analysis.thomson_kinetic_models import infer_three_dimensional_flow_from_thomson_projections

    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2**-0.5, 2**-0.5, 0.0]])
    result = infer_three_dimensional_flow_from_thomson_projections(
        [1.0e5, 2.0e5, 3.0e5], directions,
        projection_covariance_m2_s2=np.eye(3)*1.0e4,
    )
    assert result["identifiability"] == "non_identifiable"
    assert result["reportable_as_inferred_parameter"] is False
    assert result["electron_velocity_vector_m_s"] is None
    assert result["electron_velocity_covariance_m2_s2"] is None


def test_relativistic_maxwell_juttner_reports_analytic_momentum_tail_coverage():
    wavelength = np.linspace(800.0, 1328.0, 801)
    result = relativistic_maxwell_juttner_spectrum(
        wavelength, 1064.0, np.pi/2.0, 500.0,
        flow_projection_m_s=1.0e6, incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
    )
    assert result["momentum_probability_coverage"] >= 0.999
    assert result["momentum_probability_coverage"] <= 1.0
    assert result["momentum_tail_renormalization_performed"] is False
    assert result["analytic_momentum_normalization"] > 0.0
    assert result["quadrature_momentum_normalization"] > 0.0
    assert result["quadrature_to_analytic_normalization_ratio"] == pytest.approx(
        result["quadrature_momentum_normalization"]
        / result["analytic_momentum_normalization"], rel=1e-13
    )


def test_maxwell_juttner_normalization_overshoot_reports_ratio_without_unbound_variable(monkeypatch):
    import tokamak_analysis.thomson_kinetic_models as models

    monkeypatch.setattr(models, "kve", lambda order, argument: 1.0e-300)
    with pytest.raises(ValueError, match=r"analytic normalization \(captured="):
        models.relativistic_maxwell_juttner_spectrum(
            np.linspace(800.0, 1328.0, 101), 1064.0, np.pi/2.0, 500.0,
            incident_polarization=[0.0, 0.0, 1.0],
            radial_order=16, angular_order=8,
        )


def test_gyrotropic_bi_kappa_projected_marginal_has_declared_second_moment_and_coverage():
    from tokamak_analysis.thomson_kinetic_models import (
        project_gyrotropic_bi_kappa_electron_vdf,
    )

    velocity = np.linspace(-3.0e7, 3.0e7, 24001)
    b = np.array([0.0, 0.0, 1.0])
    q = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    t_parallel, t_perpendicular, kappa = 800.0, 200.0, 4.0
    flow = 1.5e5
    result = project_gyrotropic_bi_kappa_electron_vdf(
        velocity,
        temperature_parallel_eV=t_parallel,
        temperature_perpendicular_eV=t_perpendicular,
        kappa=kappa,
        magnetic_field_unit=b,
        scattering_vector_unit=q,
        flow_projection_m_s=flow,
    )
    expected_temperature = 0.5*t_parallel + 0.5*t_perpendicular
    density = result["projected_vdf_s_m"]
    area = np.trapezoid(density, velocity)
    mean = np.trapezoid(velocity*density, velocity)/area
    variance = np.trapezoid((velocity-mean)**2*density, velocity)/area

    assert result["projected_temperature_eV"] == pytest.approx(expected_temperature)
    assert result["velocity_grid_probability_coverage"] == pytest.approx(area, rel=2e-3)
    assert area > 0.999
    assert mean == pytest.approx(flow, abs=2e3)
    assert variance == pytest.approx(
        1.602176634e-19*expected_temperature/9.1093837015e-31,
        rel=3e-3,
    )
    assert result["separate_parallel_perpendicular_temperatures_identifiable"] is False


def test_gyrotropic_bi_kappa_projected_marginal_enforces_finite_variance_domain():
    from tokamak_analysis.thomson_kinetic_models import (
        project_gyrotropic_bi_kappa_electron_vdf,
    )

    with pytest.raises(ValueError, match="kappa > 3/2"):
        project_gyrotropic_bi_kappa_electron_vdf(
            np.linspace(-1e6, 1e6, 101),
            temperature_parallel_eV=100.0,
            temperature_perpendicular_eV=100.0,
            kappa=1.5,
            magnetic_field_unit=[0.0, 0.0, 1.0],
            scattering_vector_unit=[1.0, 0.0, 0.0],
        )



def test_collective_kappa_mult_population_electron_dielectric_uses_weighted_landau_responses():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 41)
    angle = np.pi/2.0
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    scattered = np.array([np.sin(angle/2), np.cos(angle/2), 0.0])
    common = dict(
        electron_density_m3=1.0e19,
        electron_temperature_eV=120.0,
        kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident,
        scattered_direction=scattered,
    )
    single = collective_kappa_spectral_density(
        wavelength, 1064.0, angle, **common
    )
    mixture = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_populations=[
            {"density_fraction": 0.85, "temperature_ratio": 0.7,
             "kappa": 5.0, "flow_projection_m_s": -1.0e5},
            {"density_fraction": 0.15, "temperature_ratio": 4.0,
             "kappa": 2.5, "flow_projection_m_s": 8.0e5},
        ],
        **common,
    )
    assert mixture["electron_population_susceptibilities"].shape == (2, wavelength.size)
    assert mixture["electron_population_projected_vdfs_s_m"].shape == (2, wavelength.size)
    np.testing.assert_allclose(
        mixture["electron_susceptibility"],
        np.sum(mixture["electron_population_susceptibilities"], axis=0),
    )
    assert not np.allclose(
        mixture["spectral_density_per_nm"], single["spectral_density_per_nm"],
        rtol=1e-4, atol=0.0,
    )
    assert sum(p["density_fraction"] for p in mixture["electron_populations"]) == pytest.approx(1.0)


def test_collective_kappa_mult_population_rejects_non_normalized_density_fractions():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi/2.0
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    scattered = np.array([np.sin(angle/2), np.cos(angle/2), 0.0])
    with pytest.raises(ValueError, match="must sum to one"):
        collective_kappa_spectral_density(
            wavelength, 1064.0, angle,
            electron_density_m3=1.0e19, electron_temperature_eV=100.0,
            kappa=3.0,
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0], ion_temperature_eV=[80.0],
            electron_velocity_m_s=[0.0, 0.0, 0.0],
            ion_velocity_m_s=[[0.0, 0.0, 0.0]],
            incident_direction=incident, scattered_direction=scattered,
            electron_populations=[
                {"density_fraction": 0.8}, {"density_fraction": 0.3}
            ],
        )


def test_collective_kappa_propagates_joint_mult_population_input_covariance(monkeypatch):
    import tokamak_analysis.thomson_kinetic_models as kinetic

    wavelength = np.linspace(1058.0, 1070.0, 41)
    populations = [
        {"density_fraction": 0.85, "temperature_ratio": 0.7,
         "kappa": 5.0, "flow_projection_m_s": -1.0e5},
        {"density_fraction": 0.15, "temperature_ratio": 4.0,
         "kappa": 2.5, "flow_projection_m_s": 8.0e5},
    ]

    def synthetic_collective_spectrum(
        wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
        electron_density_m3, electron_temperature_eV, kappa,
        electron_populations=None, **kwargs,
    ):
        x = (np.asarray(wavelength_nm) - laser_wavelength_nm) / 0.12
        if electron_populations:
            fractions = np.array([p["density_fraction"] for p in electron_populations])
            ratios = np.array([p["temperature_ratio"] for p in electron_populations])
            flows = np.array([p["flow_projection_m_s"] for p in electron_populations])
            mixture_width = np.sqrt(np.dot(fractions, ratios))
            mixture_shift = np.dot(fractions, flows) / 1.0e7
        else:
            mixture_width, mixture_shift = 1.0, 0.0
        centre = 0.04*np.log(electron_density_m3/1.0e19) + 0.02*mixture_shift
        width = np.sqrt(electron_temperature_eV/150.0)*mixture_width
        shape = np.exp(-0.5*((x-centre)/width)**2)
        shape *= 1.0 + 0.025*(kappa-3.5)*x**3/(1.0+x**4)
        return {"spectral_density_per_nm": shape}

    monkeypatch.setattr(kinetic, "collective_kappa_spectral_density", synthetic_collective_spectrum)
    common = dict(
        wavelength_nm=wavelength,
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2,
        response=np.eye(wavelength.size),
        incident_direction=[-np.sin(np.pi/4), np.cos(np.pi/4), 0.0],
        scattered_direction=[np.sin(np.pi/4), np.cos(np.pi/4), 0.0],
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        initial_density_m3=1.0e19,
        initial_temperature_eV=150.0,
        initial_kappa=3.5,
        electron_populations=populations,
    )
    nominal = synthetic_collective_spectrum(
        wavelength, 1064.0, np.pi/2, electron_density_m3=1.0e19,
        electron_temperature_eV=150.0, kappa=3.5,
        electron_populations=populations,
    )["spectral_density_per_nm"]
    signal = 2.0e4*nominal + 10.0 + 0.001*(wavelength-1064.0)
    covariance = np.diag([0.0, 1e-4, 0.0, 1e8, 0.0, 2.5e-3, 1e-4, 4e8])
    result = kinetic.fit_collective_kappa(
        signal=signal, uncertainty=np.full(wavelength.size, 0.02),
        electron_population_parameter_covariance=covariance,
        population_mc_samples=20, population_mc_seed=17,
        **common,
    )
    assert result["population_uncertainty_successful_refits"] >= 20
    assert result["electron_population_parameter_names"] == [
        "population_0_density_fraction", "population_0_temperature_ratio",
        "population_0_kappa", "population_0_flow_projection_m_s",
        "population_1_density_fraction", "population_1_temperature_ratio",
        "population_1_kappa", "population_1_flow_projection_m_s",
    ]
    assert result["population_joint_fit_input_covariance"].shape == (11, 11)
    assert result["fit_covariance"].shape == (3, 3)
    assert np.all(np.diag(result["fit_covariance"]) >=
                  np.diag(result["fit_covariance_conditional_on_population_inputs"]))



def test_collective_kappa_supports_mixed_electron_and_kappa_ion_populations():
    wavelength = np.linspace(1048.0, 1080.0, 81)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    common = dict(
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=angle,
        electron_density_m3=1.0e19,
        electron_temperature_eV=150.0,
        kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident,
        scattered_direction=scattered,
    )
    electron_mixture = [
        {"density_fraction": 0.8, "temperature_ratio": 0.7,
         "kappa": 5.0, "distribution": "kappa", "flow_projection_m_s": 0.0},
        {"density_fraction": 0.2, "temperature_ratio": 4.0,
         "distribution": "maxwellian", "flow_projection_m_s": 1.0e5},
    ]
    ion_mixture = [{"distribution": "kappa", "kappa": 3.0}]
    result = collective_kappa_spectral_density(
        wavelength, electron_populations=electron_mixture,
        ion_populations=ion_mixture, **common
    )
    assert result["electron_populations"][1]["distribution"] == "maxwellian"
    assert result["ion_populations"][0]["distribution"] == "kappa"
    assert result["ion_populations"][0]["kappa"] == pytest.approx(3.0)
    assert result["electron_population_susceptibilities"].shape == (2, wavelength.size)
    assert result["ion_susceptibility"].shape == (1, wavelength.size)
    assert np.all(np.isfinite(result["dielectric_function"]))
    assert np.all(result["spectral_density_per_nm"] >= 0.0)


def test_collective_kappa_rejects_nonphysical_ion_kappa_and_mixture_fraction():
    wavelength = np.linspace(1048.0, 1080.0, 31)
    angle = np.pi/2.0
    common = dict(
        laser_wavelength_nm=1064.0, scattering_angle_rad=angle,
        electron_density_m3=1.0e19, electron_temperature_eV=150.0, kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[-np.sin(angle/2.0), np.cos(angle/2.0), 0.0],
        scattered_direction=[np.sin(angle/2.0), np.cos(angle/2.0), 0.0],
    )
    with pytest.raises(ValueError, match="kappa must be finite and > 1.5"):
        collective_kappa_spectral_density(
            wavelength, ion_populations=[{"distribution": "kappa", "kappa": 1.4}],
            **common
        )
    with pytest.raises(ValueError, match="density_fraction values must sum to one"):
        collective_kappa_spectral_density(
            wavelength,
            electron_populations=[
                {"density_fraction": 0.7, "temperature_ratio": 1.0, "kappa": 4.0},
                {"density_fraction": 0.4, "temperature_ratio": 1.0, "kappa": 4.0},
            ],
            **common
        )


def test_gyrotropic_bimaxwellian_gls_enforces_positive_physical_temperatures():
    from tokamak_analysis.thomson_kinetic_models import infer_gyrotropic_bi_maxwellian_temperatures

    # Inconsistent noisy projections make unconstrained GLS return a negative
    # perpendicular temperature. The physical inversion must stay in T > 0.
    pitch = np.array([0.0, 0.5, 1.0])
    projected = np.array([100.0, 1.0, 1000.0])
    result = infer_gyrotropic_bi_maxwellian_temperatures(
        projected, pitch,
        projected_temperature_covariance_eV2=np.eye(3),
    )
    assert result["reportable_as_inferred_parameter"] is True
    assert result["electron_temperature_perpendicular_eV"] > 0.0
    assert result["electron_temperature_parallel_eV"] > 0.0
    assert result["temperature_constraint"] == "T_perpendicular > 0 and T_parallel > 0 eV"
    assert result["active_temperature_lower_bounds"]["T_perpendicular"] is True
    assert result["temperature_covariance_status"].startswith("local_unconstrained_GLS_curvature")


def test_gyrotropic_bimaxwellian_gls_propagates_temperature_pitch_cross_covariance():
    from tokamak_analysis.thomson_kinetic_models import infer_gyrotropic_bi_maxwellian_temperatures

    pitch = np.array([1.0, 0.5, 0.0, -0.7])
    projected = 80.0*(1.0-pitch**2) + 180.0*pitch**2
    temperature_covariance = np.eye(4)*0.04
    pitch_covariance = np.eye(4)*0.0025
    cross_covariance = np.diag([0.008, -0.006, 0.004, 0.005])
    result = infer_gyrotropic_bi_maxwellian_temperatures(
        projected, pitch,
        projected_temperature_covariance_eV2=temperature_covariance,
        magnetic_pitch_cosine_covariance=pitch_covariance,
        projected_temperature_pitch_cross_covariance=cross_covariance,
    )
    assert result["temperature_geometry_cross_covariance_included"] is True
    assert "cross_terms" in result["geometry_covariance_method"]
    derivative = 2.0*pitch*(
        result["electron_temperature_parallel_eV"]
        - result["electron_temperature_perpendicular_eV"]
    )
    expected = (
        temperature_covariance
        + derivative[:, None]*pitch_covariance*derivative[None, :]
        + cross_covariance*derivative[None, :]
        + derivative[:, None]*cross_covariance.T
    )
    np.testing.assert_allclose(
        result["effective_projected_temperature_covariance_eV2"],
        expected, rtol=2e-8, atol=1e-12,
    )


def test_gyrotropic_bimaxwellian_rejects_inconsistent_joint_temperature_geometry_covariance():
    from tokamak_analysis.thomson_kinetic_models import infer_gyrotropic_bi_maxwellian_temperatures

    with pytest.raises(ValueError, match="joint covariance block must be positive semidefinite"):
        infer_gyrotropic_bi_maxwellian_temperatures(
            [80.0, 120.0, 180.0], [0.0, 0.5, 1.0],
            projected_temperature_covariance_eV2=np.eye(3),
            magnetic_pitch_cosine_covariance=np.eye(3)*1e-4,
            projected_temperature_pitch_cross_covariance=np.eye(3)*0.1,
        )


@pytest.mark.parametrize("include_compton_recoil", [False, True])
def test_projected_relativistic_evdf_convolves_measured_laser_spectrum(include_compton_recoil):
    wavelength = np.linspace(1010.0, 1120.0, 161)
    velocity = np.linspace(-0.02, 0.02, 41) * 299_792_458.0
    laser_axis = np.linspace(1063.8, 1064.2, 9)
    laser_density = np.exp(-0.5*((laser_axis-1064.0)/0.055)**2)
    signal = np.exp(-0.5*((wavelength-1064.0)/4.0)**2)
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, signal,
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2.0,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(signal),
        velocity_grid_m_s=velocity,
        include_compton_recoil=include_compton_recoil,
        incident_polarization=[0.0, 0.0, 1.0] if include_compton_recoil else None,
        laser_spectrum_wavelength_nm=laser_axis,
        laser_spectrum_density_per_nm=laser_density,
    )
    profile = result["laser_spectral_profile"]
    assert profile["assumption"] == "measured_source_spectrum_convolved_in_exact_single_electron_kinematics"
    assert profile["captured_scattered_power_fraction_min"] >= 0.995
    assert profile["source_weighting"] == "incident_photon_flux_from_spectral_power_density"
    assert "lambda_incident/lambda_scattered" in profile["scattered_power_conversion"]
    assert result["fitted_spectrum"].shape == wavelength.shape
    assert np.all(np.isfinite(result["fitted_spectrum"]))
    assert result["recoil_included"] is include_compton_recoil
    assert result["model"] == (
        "relativistic_compton_klein_nishina_projected_1d_inverse"
        if include_compton_recoil else
        "exact_relativistic_thomson_limit_projected_1d_kinematic_inversion"
    )



def test_collective_kappa_three_population_moments_include_between_population_drift():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 41)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    populations = [
        {"density_fraction": 0.55, "temperature_ratio": 0.5, "distribution": "maxwellian", "flow_projection_m_s": -2.0e5},
        {"density_fraction": 0.30, "temperature_ratio": 2.0, "distribution": "kappa", "kappa": 4.0, "flow_projection_m_s": 1.0e5},
        {"density_fraction": 0.15, "temperature_ratio": 5.0, "distribution": "kappa", "kappa": 3.0, "flow_projection_m_s": 9.0e5},
    ]
    result = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_density_m3=1.0e19, electron_temperature_eV=200.0, kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
        electron_populations=populations,
    )
    fractions = np.array([p["density_fraction"] for p in populations])
    temperatures = 200.0*np.array([p["temperature_ratio"] for p in populations])
    flows = np.array([p["flow_projection_m_s"] for p in populations])
    mean_flow = float(fractions @ flows)
    component_variances = 1.602176634e-19*temperatures/9.1093837015e-31
    offsets = flows-mean_flow
    expected_variance = float(fractions @ (component_variances+offsets**2))
    expected_temperature = expected_variance*9.1093837015e-31/1.602176634e-19
    moments = result["electron_mixture_moments"]
    assert moments["population_count"] == 3
    assert result["electron_flow_projection_m_s"] == pytest.approx(mean_flow)
    assert moments["bulk_flow_projection_m_s"] == pytest.approx(mean_flow)
    assert moments["thermal_second_moment_temperature_eV"] == pytest.approx(float(fractions @ temperatures))
    assert moments["between_population_drift_temperature_eV"] > 0.0
    assert moments["total_second_moment_temperature_eV"] == pytest.approx(expected_temperature)
    assert result["electron_temperature_eV"] == pytest.approx(expected_temperature)
    assert moments["fourth_moment_finite"] is True
    assert moments["velocity_kurtosis"] > 3.0


def test_collective_kappa_mixture_with_low_kappa_withholds_divergent_fourth_moment():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    result = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_density_m3=1.0e19, electron_temperature_eV=200.0, kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
        electron_populations=[
            {"density_fraction": 0.8, "temperature_ratio": 1.0, "distribution": "kappa", "kappa": 2.2, "flow_projection_m_s": 0.0},
            {"density_fraction": 0.2, "temperature_ratio": 2.0, "distribution": "maxwellian", "flow_projection_m_s": 1.0e5},
        ],
    )
    moments = result["electron_mixture_moments"]
    assert moments["fourth_moment_finite"] is False
    assert moments["velocity_kurtosis"] is None
    assert moments["velocity_excess_kurtosis"] is None
    assert "1.5 < kappa <= 2.5" in moments["fourth_moment_status"]


def test_collective_kappa_single_population_moments_reduce_to_input_temperature_and_flow():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    qvector = np.asarray(scattered)-np.asarray(incident)
    input_flow = 2.5e5*qvector/np.linalg.norm(qvector)
    result = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_density_m3=1.0e19, electron_temperature_eV=200.0, kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=input_flow, ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
    )
    moments = result["electron_mixture_moments"]
    assert result["electron_flow_projection_m_s"] == pytest.approx(2.5e5)
    assert moments["total_second_moment_temperature_eV"] == pytest.approx(200.0)
    assert moments["between_population_drift_temperature_eV"] == pytest.approx(0.0, abs=1e-12)
    assert moments["velocity_kurtosis"] == pytest.approx(3.0*(2.0*4.0-3.0)/(2.0*4.0-5.0))



def test_collective_kappa_static_screening_uses_kappa_zero_frequency_susceptibility():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    ne, te, kappa, laser = 1.0e19, 200.0, 4.0, 1064.0
    result = collective_kappa_spectral_density(
        wavelength, laser, angle,
        electron_density_m3=ne, electron_temperature_eV=te, kappa=kappa,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
    )
    expected_lambda = np.sqrt(
        8.8541878128e-12*te*(2.0*kappa-3.0)/(ne*1.602176634e-19*(2.0*kappa-1.0))
    )
    expected_k = 4.0*np.pi*np.sin(angle/2.0)/(laser*1e-9)
    assert result["electron_debye_length_m"] == pytest.approx(expected_lambda, rel=1e-12)
    assert result["scattering_parameter_alpha"] == pytest.approx(1.0/(expected_k*expected_lambda), rel=1e-12)
    assert result["static_screening_status"] == "static_rest_frame_susceptibility_for_co_moving_electron_populations"


def test_collective_kappa_counterstreaming_mixture_does_not_report_scalar_debye_length():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    result = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_density_m3=1.0e19, electron_temperature_eV=200.0, kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
        electron_populations=[
            {"density_fraction": 0.7, "temperature_ratio": 0.5, "distribution": "kappa", "kappa": 4.0, "flow_projection_m_s": -2.0e5},
            {"density_fraction": 0.3, "temperature_ratio": 2.0, "distribution": "maxwellian", "flow_projection_m_s": 2.0e5},
        ],
    )
    assert result["electron_debye_length_m"] is None
    assert result["scattering_parameter_alpha"] is None
    assert result["static_screening_status"] == "not_defined_for_counter_streaming_electron_populations"


def test_collective_kappa_fit_reports_effective_mixture_temperature_and_static_alpha():
    from tokamak_analysis.thomson_kinetic_models import fit_collective_kappa, collective_kappa_spectral_density

    wavelength = np.linspace(1048.0, 1080.0, 101)
    angle = np.pi/2.0
    laser = 1064.0
    incident = np.array([-np.sin(angle/2.0), np.cos(angle/2.0), 0.0])
    scattered = np.array([np.sin(angle/2.0), np.cos(angle/2.0), 0.0])
    populations = [
        {"density_fraction": 0.7, "temperature_ratio": 0.5, "kappa": 4.0, "distribution": "kappa", "flow_projection_m_s": 0.0},
        {"density_fraction": 0.3, "temperature_ratio": 2.0, "distribution": "maxwellian", "flow_projection_m_s": 0.0},
    ]
    fixed = dict(
        electron_density_m3=1.0e19, electron_temperature_eV=150.0, kappa=3.5,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
        electron_populations=populations,
    )
    spectrum = collective_kappa_spectral_density(wavelength, laser, angle, **fixed)["spectral_density_per_nm"]
    signal = 2.0e4*spectrum + 10.0 + 0.001*(wavelength-laser)
    result = fit_collective_kappa(
        wavelength_nm=wavelength, signal=signal, laser_wavelength_nm=laser,
        scattering_angle_rad=angle, response=np.eye(wavelength.size),
        uncertainty=np.full(wavelength.size, 0.01*np.max(signal)),
        incident_direction=incident, scattered_direction=scattered,
        ion_species=fixed["ion_species"], ion_fractions=fixed["ion_fractions"],
        ion_temperature_eV=fixed["ion_temperature_eV"],
        electron_velocity_m_s=fixed["electron_velocity_m_s"],
        ion_velocity_m_s=fixed["ion_velocity_m_s"],
        initial_density_m3=1.0e19, initial_temperature_eV=150.0, initial_kappa=3.5,
        electron_populations=populations,
    )
    assert result["electron_temperature_eV"] == pytest.approx(150.0, rel=2e-3)
    assert result["electron_effective_second_moment_temperature_eV"] == pytest.approx(142.5, rel=2e-3)
    assert result["electron_velocity_projection_m_s"] == pytest.approx(0.0, abs=1e-8)
    assert result["electron_debye_length_m"] is not None
    assert result["scattering_parameter_alpha"] == pytest.approx(
        result["electron_mixture_moments"]["scattering_parameter_alpha"], rel=1e-12
    )



def test_collective_kappa_mixture_withholds_divergent_third_moment_below_kappa_two():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi/2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    result = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_density_m3=1.0e19, electron_temperature_eV=200.0, kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
        electron_populations=[
            {"density_fraction": 0.8, "temperature_ratio": 1.0, "distribution": "kappa", "kappa": 1.8, "flow_projection_m_s": -1.0e5},
            {"density_fraction": 0.2, "temperature_ratio": 2.0, "distribution": "maxwellian", "flow_projection_m_s": 2.0e5},
        ],
    )
    moments = result["electron_mixture_moments"]
    assert moments["third_moment_finite"] is False
    assert moments["third_central_velocity_moment_m3_s3"] is None
    assert moments["velocity_skewness"] is None
    assert "1.5 < kappa <= 2.0" in moments["third_moment_status"]
    assert moments["fourth_moment_finite"] is False


def test_thomson_flow_propagates_projection_geometry_cross_covariance_terms():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2**-0.5, 2**-0.5, 0.0],
    ])
    expected_velocity = np.array([1.0e5, -2.0e5, 3.0e5])
    projections = directions @ expected_velocity
    projection_covariance = np.eye(4) * 1.0e4
    geometry_covariance = np.eye(12) * 1.0e-6
    cross_covariance = np.zeros((4, 12))
    # Correlate each measured q-projection with a tangent component of its
    # own scattering-vector calibration; the joint block remains PSD.
    for view in range(4):
        cross_covariance[view, 3*view + 1] = 1.0e-2

    result = infer_three_dimensional_flow_from_thomson_projections(
        projections,
        directions,
        projection_covariance_m2_s2=projection_covariance,
        scattering_vector_covariance=geometry_covariance,
        projection_geometry_cross_covariance=cross_covariance,
    )

    assert result["reportable_as_inferred_parameter"] is True
    assert result["projection_geometry_cross_covariance_included"] is True
    assert "cross_terms" in result["geometry_covariance_method"]
    velocity = result["electron_velocity_vector_m_s"]
    jacobian = np.zeros((4, 12))
    for view, direction in enumerate(directions):
        tangent = np.eye(3) - np.outer(direction, direction)
        jacobian[view, 3*view:3*view+3] = tangent @ velocity
    tangent_projector = np.zeros((12, 12))
    for view, direction in enumerate(directions):
        tangent_projector[3*view:3*view+3, 3*view:3*view+3] = (
            np.eye(3) - np.outer(direction, direction)
        )
    tangent_cross = cross_covariance @ tangent_projector.T
    expected_effective = (
        projection_covariance
        + jacobian @ result["tangent_plane_scattering_vector_covariance"] @ jacobian.T
        + tangent_cross @ jacobian.T
        + jacobian @ tangent_cross.T
    )
    np.testing.assert_allclose(
        result["effective_projection_covariance_m2_s2"],
        expected_effective,
        rtol=2e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(velocity, expected_velocity, rtol=1e-5, atol=1e-3)


def test_thomson_flow_rejects_non_psd_joint_projection_geometry_covariance():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    directions = np.eye(3)
    with pytest.raises(ValueError, match="joint projection/geometry covariance block"):
        infer_three_dimensional_flow_from_thomson_projections(
            [1.0e5, 2.0e5, 3.0e5],
            directions,
            projection_covariance_m2_s2=np.eye(3),
            scattering_vector_covariance=np.eye(9) * 1.0e-6,
            projection_geometry_cross_covariance=np.ones((3, 9)) * 1.0,
        )


def test_thomson_three_dimensional_flow_withholds_ill_conditioned_full_rank_geometry():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    # The third view technically restores rank three, but only through a tiny
    # out-of-plane component. The inversion would amplify its noise severely.
    epsilon = 1.0e-5
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, epsilon],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    expected = np.array([1.0e5, -2.0e5, 3.0e5])
    projections = directions @ expected

    guarded = infer_three_dimensional_flow_from_thomson_projections(
        projections,
        directions,
        projection_covariance_m2_s2=np.eye(3) * 1.0e4,
    )
    assert guarded["design_rank"] == 3
    assert guarded["geometry_condition_number"] > 1.0e3
    assert guarded["geometry_condition_status"] == "ill_conditioned_vector_withheld"
    assert guarded["identifiability"] == "ill_conditioned_three_dimensional_view_geometry"
    assert guarded["reportable_as_inferred_parameter"] is False
    assert guarded["electron_velocity_vector_m_s"] is None
    assert guarded["electron_velocity_covariance_m2_s2"] is None

    explicitly_relaxed = infer_three_dimensional_flow_from_thomson_projections(
        projections,
        directions,
        projection_covariance_m2_s2=np.eye(3) * 1.0e4,
        maximum_geometry_condition_number=1.0e7,
    )
    assert explicitly_relaxed["reportable_as_inferred_parameter"] is True
    np.testing.assert_allclose(
        explicitly_relaxed["electron_velocity_vector_m_s"],
        expected,
        rtol=1.0e-8,
        atol=1.0e-3,
    )


def test_thomson_three_dimensional_flow_rejects_invalid_geometry_condition_limit():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    with pytest.raises(ValueError, match="maximum_geometry_condition_number"):
        infer_three_dimensional_flow_from_thomson_projections(
            [1.0, 2.0, 3.0],
            np.eye(3),
            projection_covariance_m2_s2=np.eye(3),
            maximum_geometry_condition_number=1.0,
        )


def test_thomson_three_dimensional_flow_withholds_vector_when_views_reject_common_flow():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2**-0.5, 2**-0.5, 0.0],
    ])
    expected = np.array([1.0e5, -2.0e5, 3.0e5])
    projections = directions @ expected
    projections[-1] += 2.0e4  # incompatible fourth view, far above stated noise

    result = infer_three_dimensional_flow_from_thomson_projections(
        projections,
        directions,
        projection_covariance_m2_s2=np.eye(4) * 1.0e4,
    )
    assert result["design_rank"] == 3
    assert result["geometry_condition_status"] == "well_conditioned"
    assert result["cross_view_consistency_p_value"] < 1.0e-3
    assert result["cross_view_consistency_status"] == "common_flow_model_rejected"
    assert result["reportable_as_inferred_parameter"] is False
    assert result["electron_velocity_vector_m_s"] is None
    assert result["electron_velocity_covariance_m2_s2"] is None


def test_thomson_three_dimensional_flow_marks_exactly_three_views_as_not_testable():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    directions = np.eye(3)
    expected = np.array([1.0e5, -2.0e5, 3.0e5])
    result = infer_three_dimensional_flow_from_thomson_projections(
        directions @ expected,
        directions,
        projection_covariance_m2_s2=np.eye(3) * 1.0e4,
    )
    assert result["cross_view_consistency_p_value"] is None
    assert result["cross_view_consistency_status"] == "not_testable_exactly_three_views"
    assert result["reportable_as_inferred_parameter"] is True


def test_thomson_three_dimensional_flow_rejects_invalid_consistency_threshold():
    from tokamak_analysis.thomson_kinetic_models import (
        infer_three_dimensional_flow_from_thomson_projections,
    )

    with pytest.raises(ValueError, match="minimum_projection_consistency_p_value"):
        infer_three_dimensional_flow_from_thomson_projections(
            [1.0, 2.0, 3.0],
            np.eye(3),
            projection_covariance_m2_s2=np.eye(3),
            minimum_projection_consistency_p_value=0.0,
        )

def test_collective_cts_bikappa_population_reduces_to_isotropic_kappa_limit():
    wavelength = np.linspace(1050.0, 1078.0, 41)
    angle = np.pi / 2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    common = dict(
        laser_wavelength_nm=1064.0, scattering_angle_rad=angle,
        electron_density_m3=1.0e19, electron_temperature_eV=120.0,
        kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
    )
    isotropic = collective_kappa_spectral_density(wavelength, **common)
    gyrotropic = collective_kappa_spectral_density(
        wavelength,
        electron_populations=[{
            "density_fraction": 1.0, "distribution": "bi_kappa",
            "temperature_parallel_eV": 120.0,
            "temperature_perpendicular_eV": 120.0,
            "magnetic_field_unit": [0.0, 0.0, 1.0],
            "kappa": 4.0, "flow_projection_m_s": 0.0,
        }],
        **common,
    )
    np.testing.assert_allclose(
        gyrotropic["electron_susceptibility"],
        isotropic["electron_susceptibility"], rtol=2e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        gyrotropic["spectral_density_per_nm"],
        isotropic["spectral_density_per_nm"], rtol=2e-10, atol=1e-12,
    )
    population = gyrotropic["electron_populations"][0]
    assert population["projected_temperature_eV"] == pytest.approx(120.0)
    assert population["magnetic_pitch_cosine"] == pytest.approx(0.0)


def test_collective_cts_bikappa_population_uses_directional_temperature_projection():
    wavelength = np.linspace(1050.0, 1078.0, 31)
    angle = np.pi / 2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    magnetic = np.array([0.0, 0.0, 1.0])
    qhat = np.array(scattered) - np.array(incident)
    qhat /= np.linalg.norm(qhat)
    mu = float(np.dot(magnetic, qhat))
    expected_tq = 90.0 * (1.0-mu**2) + 240.0 * mu**2
    result = collective_kappa_spectral_density(
        wavelength, 1064.0, angle,
        electron_density_m3=1.0e19, electron_temperature_eV=120.0, kappa=3.8,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident, scattered_direction=scattered,
        electron_populations=[{
            "density_fraction": 1.0, "distribution": "bi_kappa",
            "temperature_parallel_eV": 240.0,
            "temperature_perpendicular_eV": 90.0,
            "magnetic_field_unit": magnetic,
            "kappa": 3.8, "flow_projection_m_s": 0.0,
        }],
    )
    population = result["electron_populations"][0]
    assert population["projected_temperature_eV"] == pytest.approx(expected_tq)
    assert population["temperature_ratio"] == pytest.approx(expected_tq/120.0)
    assert population["magnetic_pitch_cosine"] == pytest.approx(mu)


def test_collective_cts_bikappa_population_rejects_missing_magnetic_geometry():
    wavelength = np.linspace(1050.0, 1078.0, 21)
    angle = np.pi / 2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    with pytest.raises(ValueError, match="magnetic_field_unit"):
        collective_kappa_spectral_density(
            wavelength, 1064.0, angle,
            electron_density_m3=1.0e19, electron_temperature_eV=120.0, kappa=3.8,
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0], ion_temperature_eV=[80.0],
            electron_velocity_m_s=[0.0, 0.0, 0.0],
            ion_velocity_m_s=[[0.0, 0.0, 0.0]],
            incident_direction=incident, scattered_direction=scattered,
            electron_populations=[{
                "density_fraction": 1.0, "distribution": "bi_kappa",
                "temperature_parallel_eV": 240.0,
                "temperature_perpendicular_eV": 90.0,
                "kappa": 3.8,
            }],
        )

def test_collective_cts_bikappa_population_is_reachable_through_public_analysis1_api():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    wavelength = np.linspace(1050.0, 1078.0, 31)
    angle = np.pi / 2.0
    incident = [-np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    scattered = [np.sin(angle/2.0), np.cos(angle/2.0), 0.0]
    result = run_advanced_thomson_physics(
        "cts_collective_kappa_spectrum",
        wavelength_nm=wavelength,
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=angle,
        electron_density_m3=1.0e19,
        electron_temperature_eV=120.0,
        kappa=3.8,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[80.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident,
        scattered_direction=scattered,
        electron_populations=[{
            "density_fraction": 1.0, "distribution": "bi_kappa",
            "temperature_parallel_eV": 240.0,
            "temperature_perpendicular_eV": 90.0,
            "magnetic_field_unit": [1.0, 0.0, 0.0],
            "kappa": 3.8, "flow_projection_m_s": 0.0,
        }],
    )
    assert result["public_physics_entrypoint"] == "cts_collective_kappa_spectrum"
    assert result["automatic_model_substitution"] is False
    assert result["electron_populations"][0]["distribution"] == "bi_kappa"
    assert result["electron_mixture_moments"]["population_count"] == 1
