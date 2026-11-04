import numpy as np
import pytest

from tokamak_analysis.thomson_kinetic_models import (
    relativistic_maxwell_juttner_spectrum,
    collective_kappa_spectral_density,
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
