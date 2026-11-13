import json

import numpy as np
import pytest

from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution


def _spectrum(sigma=0.18):
    x = np.linspace(499.0, 501.0, 501)
    y = 1000.0 * np.exp(-0.5 * ((x - 500.0) / sigma) ** 2) + 12.0 + 0.8 * (x - 500.0)
    return x, y


def _run(x, y, **kwargs):
    return analyze_thomson_spectral_distribution(
        x, y,
        instrument_response_matrix=np.eye(len(x)),
        background_fraction=0.1,
        **kwargs,
    )


def test_distribution_extracts_observed_spectral_moments_and_gaussian_shape():
    x, y = _spectrum()
    out = _run(x, y, signal_uncertainty=np.full(x.size, 1.0))
    assert out["spectral_centroid_nm"] == pytest.approx(500.0, abs=0.01)
    assert out["observed_sigma_nm"] == pytest.approx(0.18, rel=0.03)
    assert abs(out["spectral_skewness"]) < 0.03
    assert out["spectral_excess_kurtosis"] == pytest.approx(0.0, abs=0.1)
    assert out["reduced_chi_square"] is not None
    assert out["residual"].shape == y.shape


def test_distribution_requires_a_measured_response_matrix():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="measured instrument_response_matrix is required"):
        analyze_thomson_spectral_distribution(x, y, background_fraction=0.1)


def test_distribution_does_not_accept_pre_extracted_physical_parameters_or_width_corrections():
    x, y = _spectrum()
    with pytest.raises(TypeError):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(x.size),
            background_fraction=0.1,
            physical_parameters={"electron_temperature_eV": 100.0},
        )
    with pytest.raises(TypeError):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(x.size),
            background_fraction=0.1, instrument_sigma_nm=0.06,
        )


@pytest.mark.parametrize(
    "x,y,match",
    [
        (np.array([1, 2, 2, 4, 5, 6, 7, 8, 9]), np.ones(9), "strictly increasing"),
        (np.arange(10), np.ones(9), "equal lengths"),
        (np.arange(9), np.ones(9), "positive spectral area"),
    ],
)
def test_distribution_rejects_invalid_spectral_inputs(x, y, match):
    with pytest.raises(ValueError, match=match):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(len(x)),
            background_fraction=0.1,
        )


def test_distribution_rejects_nonfinite_spectral_values():
    x, y = _spectrum()
    y[20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run(x, y)


def test_distribution_rejects_invalid_response_matrix():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="finite non-negative square matrix"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.full((x.size, x.size), -1.0),
            background_fraction=0.1,
        )


def test_distribution_rejects_invalid_uncertainty_and_background_fraction():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="strictly positive"):
        _run(x, y, signal_uncertainty=np.zeros_like(y))
    with pytest.raises(ValueError, match="background_fraction"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(x.size),
            background_fraction=0.9,
        )


def test_moments_are_correct_on_nonuniform_wavelength_grid():
    left = np.linspace(499.0, 499.8, 90, endpoint=False)
    center = np.linspace(499.8, 500.2, 401, endpoint=False)
    right = np.linspace(500.2, 501.0, 90)
    x = np.unique(np.concatenate([left, center, right]))
    y = 1000.0 * np.exp(-0.5 * ((x - 500.0) / 0.18) ** 2) + 12.0
    out = _run(x, y)
    assert out["spectral_centroid_nm"] == pytest.approx(500.0, abs=0.003)
    assert out["observed_sigma_nm"] == pytest.approx(0.18, rel=0.04)
    assert out["wavelength_quadrature_weights_nm"].shape == x.shape


def test_moments_use_signed_residuals_and_propagate_pointwise_uncertainty():
    x, y = _spectrum()
    y = y.copy()
    y[0] -= 2.0
    out = analyze_thomson_spectral_distribution(
        x, y, instrument_response_matrix=np.eye(x.size),
        signal_uncertainty=np.full(x.size, 0.5),
        background_mask=(x < 499.2) | (x > 500.8),
        moment_mc_samples=40,
    )
    assert out["moment_weighting_method"] == "signed_background_subtracted_spectral_measure"
    assert out["negative_area_fraction"] > 0
    assert out["negative_area_fraction_removed_by_clipping"] == 0.0
    assert out["spectral_centroid_uncertainty_nm"] > 0
    assert out["observed_variance_uncertainty_nm2"] > 0


def test_distribution_propagates_full_within_channel_covariance():
    x, y = _spectrum()
    out = analyze_thomson_spectral_distribution(
        x, y, instrument_response_matrix=np.eye(x.size),
        background_signal=12.0 + 0.8 * (x - 500.0),
        signal_covariance=np.eye(x.size) * 0.25,
        moment_mc_samples=40,
    )
    assert out["signal_uncertainty_model"] == "full_within_channel_spectral_covariance"
    assert out["moment_uncertainty_successful_draws"] >= 20
    assert out["spectral_centroid_uncertainty_nm"] > 0


def test_distribution_propagates_joint_signal_background_cross_covariance():
    x, y = _spectrum()
    background = 12.0 + 0.8 * (x - 500.0)
    result = analyze_thomson_spectral_distribution(
        x, y, instrument_response_matrix=np.eye(x.size),
        background_signal=background,
        signal_covariance=np.eye(x.size) * 0.25,
        background_signal_covariance=np.eye(x.size) * 0.04,
        signal_background_cross_covariance=np.eye(x.size) * 0.05,
        moment_mc_samples=40,
    )
    assert result["background_uncertainty_model"] == "joint_signal_background_covariance"
    assert result["moment_uncertainty_method"] == "parametric Monte Carlo from the correlated joint signal/background block covariance"
    assert result["background_uncertainty_supplied"] is True
    assert result["moment_uncertainty_successful_draws"] >= 20
    assert result["spectral_centroid_uncertainty_nm"] > 0.0


def test_distribution_rejects_non_positive_definite_joint_signal_background_covariance():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="joint signal/background covariance"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(x.size),
            background_signal=12.0 + 0.8 * (x - 500.0),
            signal_covariance=np.eye(x.size) * 0.25,
            background_signal_covariance=np.eye(x.size) * 0.04,
            signal_background_cross_covariance=np.eye(x.size) * 0.5,
            moment_mc_samples=40,
        )


def test_distribution_rejects_conflicting_or_non_positive_definite_covariance():
    x, y = _spectrum()
    common = dict(
        instrument_response_matrix=np.eye(x.size),
        background_signal=12.0 + 0.8 * (x - 500.0),
    )
    with pytest.raises(ValueError, match="not both"):
        analyze_thomson_spectral_distribution(
            x, y, **common, signal_uncertainty=np.ones(x.size),
            signal_covariance=np.eye(x.size),
        )
    with pytest.raises(ValueError, match="positive definite"):
        analyze_thomson_spectral_distribution(
            x, y, **common, signal_covariance=np.ones((x.size, x.size)),
        )


def test_distribution_requires_one_explicit_background_specification():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="Background is not inferred automatically"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(x.size),
        )


def test_distribution_rejects_multiple_background_specifications():
    x, y = _spectrum()
    mask = np.zeros(x.size, dtype=bool)
    mask[:20] = True
    with pytest.raises(ValueError, match="exactly one background specification"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=np.eye(x.size),
            background_mask=mask, background_fraction=0.1,
        )


def test_distribution_does_not_return_instrument_corrected_width_or_injected_physics():
    x, y = _spectrum()
    out = _run(x, y)
    forbidden = {
        "instrument_corrected_sigma_nm",
        "instrument_corrected_variance_nm2",
        "instrument_corrected_sigma_uncertainty_nm",
        "physical_parameter_extraction",
    }
    assert forbidden.isdisjoint(out)


def test_distribution_result_is_strict_json_safe():
    x, y = _spectrum()
    out = _run(x, y, signal_uncertainty=np.full(x.size, 1.0))
    payload = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in out.items()
    }
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert len(decoded["wavelength_nm"]) == x.size
    assert decoded["gaussian_fit_parameters"]["center_nm"] == pytest.approx(500.0, abs=0.01)


def test_distribution_is_invariant_to_signal_units():
    x, y = _spectrum()
    baseline = _run(x, y)
    scaled = _run(x, y * 1e-12)
    assert scaled["spectral_centroid_nm"] == pytest.approx(baseline["spectral_centroid_nm"], abs=1e-8)
    assert scaled["observed_sigma_nm"] == pytest.approx(baseline["observed_sigma_nm"], rel=1e-8)
    assert scaled["gaussian_fit_parameters"]["amplitude"] == pytest.approx(
        baseline["gaussian_fit_parameters"]["amplitude"] * 1e-12, rel=1e-6
    )


def test_distribution_reports_optimizer_failure_as_actionable_value_error(monkeypatch):
    import tokamak_analysis.thomson_distribution as module

    x, y = _spectrum()

    def fail_fit(*args, **kwargs):
        raise RuntimeError("synthetic optimizer non-convergence")

    monkeypatch.setattr(module, "curve_fit", fail_fit)
    with pytest.raises(ValueError, match="Gaussian spectral-shape fit failed"):
        _run(x, y)


def test_thomson_distribution_is_independently_registered():
    from tokamak_analysis.analysis_registry import ANALYSES, get_analysis

    assert len(ANALYSES) >= 81
    assert get_analysis(81)["name"] == "Thomson spectral distribution"
    assert get_analysis(81)["function"] == "analyze_thomson_spectral_distribution"
    input_names = {
        item["name"]
        for item in get_analysis(81)["execution_contract"]["input_specs"]
    }
    assert "fit_geometry_cross_covariance_nm2_rad" in input_names
    output_names = {
        item["name"]
        for item in get_analysis(81)["execution_contract"]["output_specs"]
    }
    assert "projected_evdf_physical_moments" in output_names


def test_distribution_exposes_background_and_statistical_interpretation():
    x, y = _spectrum()
    weighted = _run(x, y, signal_uncertainty=np.full(x.size, 1.0))
    assert weighted["background_identifiability_status"] == "edge_fraction_assumption_not_independently_identified"
    assert weighted["gaussian_fit_statistical_weighting_status"] == "weighted_by_independent_pointwise_measurement_uncertainty"
    assert "negative residuals" in weighted["moment_interpretation"]
    unweighted = _run(x, y)
    assert unweighted["gaussian_fit_statistical_weighting_status"] == "unweighted_least_squares_no_measurement_uncertainty_supplied"
    assert unweighted["gaussian_fit_covariance_interpretation"] == "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"


def test_explicit_background_uncertainty_is_propagated_into_moment_uncertainty():
    x = np.linspace(500.0, 564.0, 161)
    signal = 100.0 * np.exp(-0.5 * ((x - 532.0) / 7.0) ** 2) + 12.0
    background = np.full_like(x, 12.0)
    background_covariance = np.eye(x.size) * 0.04
    result = analyze_thomson_spectral_distribution(
        x, signal, instrument_response_matrix=np.eye(x.size),
        background_signal=background,
        background_signal_covariance=background_covariance,
        moment_mc_samples=40,
    )
    assert result["background_uncertainty_supplied"] is True
    assert result["background_uncertainty_model"] == "independent_full_background_signal_covariance"
    assert result["moment_uncertainty_successful_draws"] >= 20
    assert result["spectral_centroid_uncertainty_nm"] > 0.0


def test_background_uncertainty_requires_explicit_background_signal():
    x = np.linspace(500.0, 564.0, 81)
    signal = 100.0 * np.exp(-0.5 * ((x - 532.0) / 7.0) ** 2) + 12.0
    with pytest.raises(ValueError, match="only with an explicitly supplied background_signal"):
        analyze_thomson_spectral_distribution(
            x, signal, instrument_response_matrix=np.eye(x.size),
            background_fraction=0.1,
            background_signal_uncertainty=np.full(x.size, 0.1),
        )


def test_distribution_rejects_non_normalized_measured_response_without_modifying_it():
    x, y = _spectrum()
    response = np.eye(x.size)
    response[0, 0] = 0.5
    original = response.copy()
    with pytest.raises(ValueError, match="columns must already sum to one"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_response_matrix=response, background_fraction=0.1,
        )
    np.testing.assert_array_equal(response, original)


def test_nonuniform_wavelength_moments_use_one_finite_volume_measure():
    from tokamak_analysis.thomson_distribution import (
        _moment_statistics,
        _wavelength_bin_widths,
    )

    wavelength = np.array([0.0, 0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0])
    signal = np.array([0.2, 0.8, 2.0, 4.0, 3.0, 1.5, 0.7, 0.3, 0.1])
    background = np.zeros_like(signal)
    result = _moment_statistics(
        wavelength,
        signal,
        background_fraction=None,
        background_mask=None,
        background_signal=background,
    )

    widths = _wavelength_bin_widths(wavelength)
    expected_area = float(np.sum(signal * widths))
    expected_center = float(np.sum(wavelength * signal * widths) / expected_area)
    expected_variance = float(
        np.sum((wavelength - expected_center) ** 2 * signal * widths)
        / expected_area
    )
    assert result["signed_area"] == pytest.approx(expected_area, rel=1e-13)
    assert result["positive_area"] == pytest.approx(expected_area, rel=1e-13)
    assert result["center"] == pytest.approx(expected_center, rel=1e-13)
    assert result["variance"] == pytest.approx(expected_variance, rel=1e-13)



def test_spectral_distribution_marks_gaussian_width_as_nonphysical_temperature():
    import numpy as np
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.0, 1061.0, 41)
    signal = 2.0 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.2) ** 2) + 0.05
    response = np.eye(wavelength.size)
    result = analyze_thomson_spectral_distribution(
        wavelength, signal, instrument_response_matrix=response,
        background_fraction=0.1,
    )
    assert result["gaussian_width_is_physical_temperature"] is False
    assert "not a unique electron temperature" in result["gaussian_width_interpretation"]
    assert "canonical Thomson physical forward-model fit" in result["physical_temperature_source"]


def test_selected_maxwellian_maps_intrinsic_gaussian_width_to_conditional_physics():
    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    angle = np.pi / 2.0
    q = 2.0 * np.sin(angle / 2.0)
    temperature = 120.0
    sigma = laser * q * np.sqrt(temperature / 510998.95069)
    center = laser * (1.0 - q * 2.0e4 / 299792458.0)
    signal = 5.0 * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2) + 0.2
    result = analyze_thomson_spectral_distribution(
        wavelength, signal, instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.2),
        signal_uncertainty=np.full(wavelength.size, 0.01),
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=laser, scattering_angle_rad=angle,
    )
    physical = result["physical_inference"]
    assert physical["status"] == "conditional_gaussian_maxwellian_candidate"
    assert physical["reportable_as_primary_inference"] is False
    assert physical["electron_temperature_candidate_eV"] == pytest.approx(temperature, rel=2e-3)
    assert physical["electron_velocity_projection_candidate_m_s"] == pytest.approx(2.0e4, rel=2e-3)
    assert physical["parameter_covariance"].shape == (2, 2)
    assert physical["parameter_standard_uncertainties"]["electron_temperature_eV"] > 0.0


def test_maxwellian_physical_covariance_includes_fit_geometry_cross_terms():
    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    angle = np.pi / 2.0
    q = 2.0 * np.sin(angle / 2.0)
    temperature = 120.0
    sigma = laser * q * np.sqrt(temperature / 510998.95069)
    center = laser * (1.0 - q * 2.0e4 / 299792458.0)
    signal = 5.0 * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2) + 0.2
    result = analyze_thomson_spectral_distribution(
        wavelength, signal, instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.2),
        signal_uncertainty=np.full(wavelength.size, 0.01),
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=laser, scattering_angle_rad=angle,
        laser_wavelength_uncertainty_nm=1.0e-2,
        scattering_angle_uncertainty_rad=1.0e-2,
        fit_geometry_cross_covariance_nm2_rad=np.array([
            [1.0e-10, 1.0e-10],
            [1.0e-10, 1.0e-10],
        ]),
    )
    physical = result["physical_inference"]
    assert physical["parameter_covariance"].shape == (2, 2)
    assert np.all(np.isfinite(physical["parameter_covariance"]))
    assert physical["parameter_covariance"][0, 1] == pytest.approx(
        physical["parameter_covariance"][1, 0], rel=1e-12, abs=1e-14
    )


def test_maxwellian_physical_covariance_rejects_inconsistent_fit_geometry_cross_terms():
    wavelength = np.linspace(520.0, 544.0, 241)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 1.0) ** 2) + 0.2
    with pytest.raises(ValueError, match="joint fitted-spectrum/laser/angle covariance"):
        analyze_thomson_spectral_distribution(
            wavelength, signal, instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(wavelength, 0.2),
            physical_model="non_collective_maxwellian",
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
            fit_geometry_cross_covariance_nm2_rad=np.ones((2, 2)),
        )


def test_selected_maxwellian_physical_conversion_rejects_relativistic_domain():
    wavelength = np.linspace(500.0, 564.0, 161)
    signal = 3.0 * np.exp(-0.5 * ((wavelength - 532.0) / 5.0) ** 2) + 0.1
    result = analyze_thomson_spectral_distribution(
        wavelength, signal, instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.1),
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
    )
    assert result["physical_inference"]["status"] == "rejected_outside_declared_nonrelativistic_validity_domain"
    assert result["physical_inference"]["electron_temperature_candidate_eV"] is None


def test_spectral_distribution_requires_explicit_physical_model_geometry():
    wavelength = np.linspace(520.0, 544.0, 121)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2) + 0.1
    with pytest.raises(ValueError, match="laser_wavelength_nm and scattering_angle_rad"):
        analyze_thomson_spectral_distribution(
            wavelength, signal, instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(wavelength, 0.1),
            physical_model="non_collective_maxwellian",
        )


def test_spectral_distribution_dispatches_kappa_to_native_physical_forward_fit(monkeypatch):
    import tokamak_analysis.thomson_kinetic_background as kinetic_background
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    response = np.eye(wavelength.size)
    captured = {}

    def fake_fit(x, y, **kwargs):
        captured.update(kwargs)
        return {
            "status": "identified_from_measurement",
            "primary_result_status": "identified_from_measurement",
            "parameter_estimates": {"electron_temperature_eV": 120.0, "kappa": 4.0},
        }

    monkeypatch.setattr(kinetic_background, "fit_noncollective_kinetic_thomson_with_background", fake_fit)
    result = analyze_thomson_spectral_distribution(
        wavelength, signal,
        instrument_response_matrix=response,
        background_signal=np.full_like(signal, 0.01),
        physical_model="kappa",
        laser_wavelength_nm=1060.0,
        scattering_angle_rad=np.pi / 2,
        physical_fit_options={
            "physical_initial": [120.0, 0.0, 4.0],
            "background_model": "linear",
        },
    )

    assert captured["physical_model"] == "kappa"
    assert captured["instrument_response"] is response
    assert result["physical_inference"]["gaussian_width_used_as_temperature"] is False
    assert result["physical_inference"]["reportable_as_primary_inference"] is True


def test_spectral_distribution_registry_inputs_reach_native_kinetic_solver(monkeypatch):
    import tokamak_analysis.thomson_kinetic_background as kinetic_background
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    response = np.eye(wavelength.size)
    discrepancy = np.eye(wavelength.size) * 1e-6
    systematic = np.eye(wavelength.size) * 2e-6
    polarization = np.array([0.0, 0.0, 1.0])
    captured = {}

    def fake_fit(x, y, **kwargs):
        captured.update(kwargs)
        return {
            "status": "identified_from_measurement",
            "primary_result_status": "identified_from_measurement",
            "parameter_estimates": {"electron_temperature_eV": 120.0, "kappa": 4.0},
        }

    monkeypatch.setattr(kinetic_background, "fit_noncollective_kinetic_thomson_with_background", fake_fit)
    result = analyze_thomson_spectral_distribution(
        wavelength, signal,
        instrument_response_matrix=response,
        background_signal=np.full_like(signal, 0.01),
        physical_model="kappa",
        laser_wavelength_nm=1060.0,
        scattering_angle_rad=np.pi / 2,
        physical_initial=[120.0, 0.0, 4.0],
        incident_polarization=polarization,
        include_compton_recoil=True,
        physical_background_model="quadratic",
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
    )

    assert captured["physical_initial"] == [120.0, 0.0, 4.0]
    assert captured["incident_polarization"] is polarization
    assert captured["include_compton_recoil"] is True
    assert captured["background_model"] == "quadratic"
    assert captured["model_discrepancy_covariance"] is discrepancy
    assert captured["systematic_covariance"] is systematic
    assert captured["instrument_response"] is response
    assert result["physical_inference"]["reportable_as_primary_inference"] is True


def test_spectral_distribution_rejects_duplicate_top_level_and_nested_solver_controls():
    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    with pytest.raises(ValueError, match="either as a top-level argument"):
        analyze_thomson_spectral_distribution(
            wavelength, signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(signal, 0.01),
            physical_model="kappa",
            laser_wavelength_nm=1060.0,
            scattering_angle_rad=np.pi / 2,
            physical_initial=[120.0, 0.0, 4.0],
            physical_fit_options={"physical_initial": [120.0, 0.0, 4.0]},
        )


def test_spectral_distribution_rejects_kinetic_model_without_explicit_fit_options():
    import pytest
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    with pytest.raises(ValueError, match="physical_fit_options"):
        analyze_thomson_spectral_distribution(
            wavelength, signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(signal, 0.01),
            physical_model="kappa",
            laser_wavelength_nm=1060.0,
            scattering_angle_rad=np.pi / 2,
        )



def test_selected_maxwellian_can_dispatch_to_canonical_measured_spectrum_inference(monkeypatch):
    import tokamak_analysis.thomson_inference as inference
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    response = np.eye(wavelength.size)
    captured = {}

    def fake_canonical(x, spectra, radius, **kwargs):
        captured.update({
            "wavelength": np.asarray(x).copy(),
            "spectra": np.asarray(spectra).copy(),
            "radius": np.asarray(radius).copy(),
            "kwargs": kwargs,
        })
        return {
            "channels": [{
                "electron_temperature_eV": 125.0,
                "electron_velocity_projection_m_s": 12000.0,
            }],
            "result_classification": {
                "channels": [{
                    "primary": [
                        {"parameter": "electron_temperature_eV"},
                        {"parameter": "electron_velocity_projection_m_s"},
                    ]
                }]
            },
        }

    monkeypatch.setattr(inference, "analyze_thomson_spectra", fake_canonical)
    result = analyze_thomson_spectral_distribution(
        wavelength, signal,
        instrument_response_matrix=response,
        background_signal=np.full_like(signal, 0.01),
        signal_uncertainty=np.full_like(signal, 0.002),
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=1060.0,
        scattering_angle_rad=np.pi / 2,
        physical_fit_options={"radius_m": 0.42, "background_model": "polynomial", "background_degree": 2},
    )

    physical = result["physical_inference"]
    assert physical["status"] == "identified_from_measured_spectrum"
    assert physical["reportable_as_primary_inference"] is True
    assert physical["gaussian_width_used_as_temperature"] is False
    assert physical["gaussian_width_role"] == "independent_descriptive_cross_check_only"
    np.testing.assert_array_equal(captured["wavelength"], wavelength)
    np.testing.assert_array_equal(captured["spectra"], signal.reshape(1, -1))
    np.testing.assert_array_equal(captured["radius"], np.array([0.42]))
    assert captured["kwargs"]["model"] == "non_collective_maxwellian"
    assert captured["kwargs"]["instrument_response"] is response
    assert captured["kwargs"]["signal_uncertainty"].shape == (1, wavelength.size)
    assert captured["kwargs"]["background_model"] == "polynomial"


def test_spectral_distribution_top_level_controls_reach_canonical_optical_fit(monkeypatch):
    import tokamak_analysis.thomson_inference as inference
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    response = np.eye(wavelength.size)
    discrepancy = np.eye(wavelength.size) * 1e-6
    systematic = np.eye(wavelength.size) * 2e-6
    polarization = np.array([0.0, 0.0, 1.0])
    captured = {}

    def fake_canonical(x, spectra, radius, **kwargs):
        captured.update(kwargs)
        return {
            "channels": [{}],
            "result_classification": {
                "channels": [{
                    "primary": [
                        {"parameter": "electron_temperature_eV"},
                        {"parameter": "electron_velocity_projection_m_s"},
                    ]
                }]
            },
        }

    monkeypatch.setattr(inference, "analyze_thomson_spectra", fake_canonical)
    analyze_thomson_spectral_distribution(
        wavelength, signal,
        instrument_response_matrix=response,
        background_signal=np.full_like(signal, 0.01),
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=1060.0,
        scattering_angle_rad=np.pi / 2,
        physical_background_model="polynomial",
        incident_polarization=polarization,
        include_compton_recoil=True,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
        physical_fit_options={"radius_m": 0.42, "background_degree": 2},
    )

    assert captured["background_model"] == "polynomial"
    assert captured["incident_polarization"] is polarization
    assert captured["include_compton_recoil"] is True
    assert captured["model_discrepancy_covariance"] is discrepancy
    assert captured["systematic_covariance"] is systematic


def test_maxwellian_canonical_inference_requires_radius_in_fit_options():
    import pytest
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    with pytest.raises(ValueError, match="must include radius_m"):
        analyze_thomson_spectral_distribution(
            wavelength, signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(signal, 0.01),
            physical_model="non_collective_maxwellian",
            laser_wavelength_nm=1060.0,
            scattering_angle_rad=np.pi / 2,
            physical_fit_options={"background_model": "linear"},
        )


@pytest.mark.parametrize("uncertainty_kind", ["pointwise", "full_covariance"])
def test_kinetic_physical_fit_receives_top_level_spectral_measurement_uncertainty(
    monkeypatch, uncertainty_kind
):
    import tokamak_analysis.thomson_kinetic_background as kinetic_background

    x, y = _spectrum()
    captured = {}

    def fake_kinetic_fit(wavelength, measured, **kwargs):
        captured["wavelength"] = np.asarray(wavelength).copy()
        captured["measured"] = np.asarray(measured).copy()
        captured["kwargs"] = kwargs
        return {
            "status": "completed",
            "primary_result_status": "identified_from_measurement",
            "parameter_order": ["electron_temperature_eV", "electron_velocity_projection_m_s", "kappa"],
        }

    monkeypatch.setattr(
        kinetic_background,
        "fit_noncollective_kinetic_thomson_with_background",
        fake_kinetic_fit,
    )
    supplied = (
        np.full(x.size, 0.75)
        if uncertainty_kind == "pointwise"
        else np.diag(np.full(x.size, 0.75**2))
    )
    uncertainty_kwargs = (
        {"signal_uncertainty": supplied}
        if uncertainty_kind == "pointwise"
        else {"signal_covariance": supplied}
    )

    result = analyze_thomson_spectral_distribution(
        x,
        y,
        instrument_response_matrix=np.eye(x.size),
        background_fraction=0.1,
        physical_model="kappa",
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2,
        physical_fit_options={"physical_initial": [120.0, 0.0, 4.0]},
        **uncertainty_kwargs,
    )

    np.testing.assert_array_equal(captured["wavelength"], x)
    np.testing.assert_array_equal(captured["measured"], y)
    if uncertainty_kind == "pointwise":
        np.testing.assert_array_equal(
            captured["kwargs"]["measurement_uncertainty"], supplied
        )
        assert "measurement_covariance" not in captured["kwargs"]
    else:
        np.testing.assert_array_equal(
            captured["kwargs"]["measurement_covariance"], supplied
        )
        assert "measurement_uncertainty" not in captured["kwargs"]
    assert result["physical_inference"]["reportable_as_primary_inference"] is True


def test_kinetic_physical_fit_rejects_duplicate_top_level_and_nested_uncertainty(
    monkeypatch,
):
    import tokamak_analysis.thomson_kinetic_background as kinetic_background

    x, y = _spectrum()

    def unused_fit(*args, **kwargs):
        raise AssertionError("the conflicting fit must be rejected before solver execution")

    monkeypatch.setattr(
        kinetic_background,
        "fit_noncollective_kinetic_thomson_with_background",
        unused_fit,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        analyze_thomson_spectral_distribution(
            x,
            y,
            instrument_response_matrix=np.eye(x.size),
            background_fraction=0.1,
            signal_uncertainty=np.full(x.size, 0.75),
            physical_model="kappa",
            laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2,
            physical_fit_options={
                "physical_initial": [120.0, 0.0, 4.0],
                "measurement_uncertainty": np.full(x.size, 0.5),
            },
        )



def test_maxwellian_geometry_cross_covariance_is_propagated_into_physical_covariance():
    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    angle = np.pi / 2.0
    q = 2.0 * np.sin(angle / 2.0)
    temperature = 120.0
    sigma = laser * q * np.sqrt(temperature / 510998.95069)
    center = laser * (1.0 - q * 2.0e4 / 299792458.0)
    signal = 5.0 * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2) + 0.2
    common = dict(
        wavelength_nm=wavelength,
        signal=signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.2),
        signal_uncertainty=np.full(wavelength.size, 0.01),
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        laser_wavelength_uncertainty_nm=0.01,
        scattering_angle_uncertainty_rad=0.01,
    )
    independent = analyze_thomson_spectral_distribution(**common)
    correlated = analyze_thomson_spectral_distribution(
        **common, laser_angle_covariance_nm_rad=5.0e-5
    )
    assert independent["physical_inference"]["parameter_covariance"] is not None
    assert correlated["physical_inference"]["parameter_covariance"] is not None
    assert not np.allclose(
        independent["physical_inference"]["parameter_covariance"],
        correlated["physical_inference"]["parameter_covariance"],
        rtol=1e-8, atol=1e-14,
    )


def test_maxwellian_geometry_cross_covariance_must_be_positive_semidefinite():
    wavelength = np.linspace(520.0, 544.0, 121)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2) + 0.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        analyze_thomson_spectral_distribution(
            wavelength, signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(wavelength, 0.1),
            physical_model="non_collective_maxwellian",
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0,
            laser_wavelength_uncertainty_nm=0.01,
            scattering_angle_uncertainty_rad=0.01,
            laser_angle_covariance_nm_rad=2.0e-4,
        )


def test_spectral_distribution_connects_same_spectrum_to_projected_evdf():
    wavelength = np.linspace(520.0, 544.0, 241)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 1.0) ** 2) + 0.02
    result = analyze_thomson_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.02),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        signal_uncertainty=np.full_like(wavelength, 1e-3),
        projected_evdf_options={
            "velocity_limit_m_s": 1.0e6,
            "velocity_points": 81,
            "regularization_strength": 1e-3,
        },
    )
    inverse = result["projected_evdf_inference"]
    assert inverse["source_spectrum"] == "same_calibrated_measured_spectrum_as_spectral_distribution"
    assert inverse["inference_scope"] == "one_dimensional_q_parallel_velocity_marginal_only"
    assert inverse["gaussian_width_used_as_evdf"] is False
    assert np.isclose(inverse["evdf_normalization"], 1.0, rtol=1e-6, atol=1e-8)
    assert inverse["velocity_parallel_m_s"].shape == (81,)
    assert inverse["evdf_parallel_s_per_m"].shape == (81,)
    selection = inverse["parametric_model_selection_uncertainty"]
    assert selection["status"] == "akaike_weighted_model_uncertainty"
    assert sum(selection["weights"].values()) == pytest.approx(1.0, abs=1e-12)
    closure = inverse["spectral_moment_closure"]
    assert closure["comparison_performed"] is True
    assert closure["status"] == "conditional_moment_closure"
    assert np.isfinite(closure["centroid_closure_residual_nm"])
    assert np.isfinite(closure["variance_closure_residual_nm2"])

    # The same additive model/systematic covariance must reach the EVDF inverse,
    # rather than silently disappearing at the spectral-distribution wrapper.
    discrepancy = np.eye(wavelength.size) * 2.0e-7
    systematic = np.eye(wavelength.size) * 1.0e-7
    propagated = analyze_thomson_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.02),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        signal_uncertainty=np.full_like(wavelength, 1e-3),
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
        projected_evdf_options={
            "velocity_limit_m_s": 1.0e6,
            "velocity_points": 81,
            "regularization_strength": 1e-3,
        },
    )["projected_evdf_inference"]
    assert propagated["model_discrepancy_covariance_included"] is True
    assert propagated["systematic_covariance_included"] is True


def test_model_discrepancy_and_systematic_covariance_propagate_to_spectral_moments():
    wavelength = np.linspace(499.0, 501.0, 101)
    signal = 4.0 * np.exp(-0.5 * ((wavelength - 500.0) / 0.16) ** 2) + 0.2
    common = dict(
        wavelength_nm=wavelength,
        signal=signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.2),
        signal_uncertainty=np.full_like(wavelength, 0.002),
        moment_mc_samples=100,
    )
    baseline = analyze_thomson_spectral_distribution(**common)
    discrepancy = np.eye(wavelength.size) * 4.0e-4
    systematic = np.eye(wavelength.size) * 2.0e-4
    propagated = analyze_thomson_spectral_distribution(
        **common,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
    )
    assert propagated["moment_uncertainty_covariance_components"] == {
        "measurement_covariance_included": True,
        "background_covariance_included": False,
        "model_discrepancy_covariance_included": True,
        "systematic_covariance_included": True,
        "components_assumed_mutually_independent": True,
        "sampling_method": "independent_multivariate_gaussian_components_within_component_correlations_preserved",
    }
    assert propagated["moment_uncertainty_successful_draws"] >= 80
    assert propagated["observed_sigma_uncertainty_nm"] > baseline["observed_sigma_uncertainty_nm"]


def test_model_discrepancy_covariance_rejects_non_psd_matrix():
    wavelength = np.linspace(499.0, 501.0, 31)
    signal = np.exp(-0.5 * ((wavelength - 500.0) / 0.2) ** 2) + 0.1
    invalid = np.eye(wavelength.size)
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="model_discrepancy_covariance must be positive semidefinite"):
        analyze_thomson_spectral_distribution(
            wavelength, signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(wavelength, 0.1),
            model_discrepancy_covariance=invalid,
        )


def test_model_discrepancy_covariance_psd_check_is_relative_to_covariance_scale():
    wavelength = np.linspace(499.0, 501.0, 31)
    signal = np.exp(-0.5 * ((wavelength - 500.0) / 0.2) ** 2) + 0.1
    invalid = np.eye(wavelength.size) * 1.0e-12
    invalid[0, 0] = -1.0e-12
    with pytest.raises(ValueError, match="model_discrepancy_covariance must be positive semidefinite"):
        analyze_thomson_spectral_distribution(
            wavelength, signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(wavelength, 0.1),
            model_discrepancy_covariance=invalid,
        )


def test_spectral_distribution_dispatches_compton_to_exact_relativistic_evdf(monkeypatch):
    import tokamak_analysis.thomson_gap_closure as relativistic

    wavelength = np.linspace(520.0, 544.0, 81)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 1.2) ** 2) + 0.02
    captured = {}

    def fake_relativistic(**kwargs):
        captured.update(kwargs)
        velocity = np.linspace(-1.0e6, 1.0e6, 81)
        pdf = np.exp(-0.5 * (velocity / 2.0e5) ** 2)
        pdf /= np.trapz(pdf, velocity)
        scattered = np.exp(-0.5 * ((wavelength - 532.0) / 1.2) ** 2)
        return {
            "velocity_grid_m_s": velocity,
            "projected_evdf_s_m": pdf,
            "fitted_scattered_spectrum": scattered,
            "fitted_spectrum": scattered + 0.02,
        }

    monkeypatch.setattr(relativistic, "reconstruct_relativistic_projected_thomson_evdf", fake_relativistic)
    discrepancy = np.eye(wavelength.size) * 1.0e-6
    systematic = np.eye(wavelength.size) * 2.0e-6
    result = analyze_thomson_spectral_distribution(
        wavelength, signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.02),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        incident_polarization=[0.0, 0.0, 1.0],
        include_compton_recoil=True,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
        projected_evdf_options={
            "kinematics": "compton", "velocity_limit_m_s": 1.0e6,
            "velocity_points": 81, "regularization_strength": 1.0e-3,
        },
    )
    assert captured["include_compton_recoil"] is True
    assert captured["incident_polarization"] == [0.0, 0.0, 1.0]
    np.testing.assert_allclose(captured["systematic_covariance"], systematic)
    assert result["projected_evdf_inference"]["kinematic_model"] == "exact_relativistic_compton"
    assert result["projected_evdf_inference"]["spectral_moment_closure"]["status"] == "relativistic_response_aware_moment_closure"


def test_projected_evdf_connection_requires_explicit_scattering_geometry():
    wavelength = np.linspace(520.0, 544.0, 121)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2) + 0.1
    with pytest.raises(ValueError, match="laser_wavelength_nm and scattering_angle_rad"):
        analyze_thomson_spectral_distribution(
            wavelength,
            signal,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.full_like(wavelength, 0.1),
            projected_evdf_options={"velocity_points": 81},
        )


@pytest.mark.parametrize("selected_model", ["collective_maxwellian", "collective_kappa"])
def test_spectral_distribution_dispatches_selected_collective_physical_model(
    monkeypatch, selected_model
):
    import tokamak_analysis.thomson_inference as inference

    wavelength, signal = _spectrum()
    captured = {}

    def fake_canonical_fit(wavelength_nm, spectra, radius_m, **kwargs):
        captured["wavelength"] = np.asarray(wavelength_nm).copy()
        captured["spectra"] = np.asarray(spectra).copy()
        captured["radius"] = np.asarray(radius_m).copy()
        captured["kwargs"] = kwargs
        return {
            "channels": [{"electron_density_m3": 2.0e19,
                          "electron_temperature_eV": 150.0}],
            "result_classification": {
                "channels": [{
                    "primary": [
                        {"parameter": "electron_density_m3"},
                        {"parameter": "electron_temperature_eV"},
                        {"parameter": "electron_velocity_projection_m_s"},
                    ]
                }]
            },
        }

    monkeypatch.setattr(inference, "analyze_thomson_spectra", fake_canonical_fit)
    uncertainty = np.full(wavelength.size, 0.02)
    result = analyze_thomson_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_fraction=0.1,
        signal_uncertainty=uncertainty,
        physical_model=selected_model,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        physical_fit_options={
            "radius_m": 0.42,
            "ion_temperature_eV": 80.0,
            "ion_species": [{"charge": 1, "mass_amu": 2.0, "density_fraction": 1.0}],
        },
    )

    assert captured["kwargs"]["model"] == selected_model
    assert captured["kwargs"]["ion_temperature_eV"] == 80.0
    np.testing.assert_array_equal(captured["wavelength"], wavelength)
    np.testing.assert_array_equal(captured["spectra"], signal.reshape(1, -1))
    np.testing.assert_array_equal(captured["radius"], [0.42])
    np.testing.assert_array_equal(captured["kwargs"]["signal_uncertainty"], uncertainty.reshape(1, -1))
    assert result["physical_inference"]["selected_model"] == selected_model
    assert result["physical_inference"]["reportable_as_primary_inference"] is True
    assert result["physical_inference"]["forward_model"] == (
        "canonical_optical_thomson_forward_fit:" + selected_model
    )


@pytest.mark.parametrize(
    ("public_model", "solver_model", "initial"),
    [
        ("non_collective_kappa", "kappa", [120.0, 0.0, 4.0]),
        (
            "non_collective_relativistic_maxwell_juttner",
            "relativistic_maxwell_juttner",
            [120.0, 0.0],
        ),
    ],
)
def test_spectral_distribution_accepts_canonical_analysis1_kinetic_model_ids(
    monkeypatch, public_model, solver_model, initial
):
    import tokamak_analysis.thomson_kinetic_background as kinetic_background

    wavelength = np.linspace(1059.8, 1060.2, 61)
    signal = 1.0 + 0.2 * np.exp(-0.5 * ((wavelength - 1060.0) / 0.045) ** 2)
    captured = {}

    def fake_fit(x, y, **kwargs):
        captured.update(kwargs)
        return {
            "status": "identified_from_measurement",
            "primary_result_status": "identified_from_measurement",
            "parameter_estimates": {"electron_temperature_eV": 120.0},
        }

    monkeypatch.setattr(
        kinetic_background,
        "fit_noncollective_kinetic_thomson_with_background",
        fake_fit,
    )
    result = analyze_thomson_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(signal, 0.01),
        physical_model=public_model,
        laser_wavelength_nm=1060.0,
        scattering_angle_rad=np.pi / 2.0,
        physical_fit_options={"physical_initial": initial, "background_model": "linear"},
    )

    assert captured["physical_model"] == solver_model
    assert result["physical_inference"]["selected_model"] == public_model
    assert result["physical_inference"]["solver_model"] == solver_model
    assert result["physical_inference"]["gaussian_width_used_as_temperature"] is False
    assert result["physical_inference"]["reportable_as_primary_inference"] is True


def test_spectral_distribution_kappa_inference_is_reportable_from_real_joint_fit():
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    wavelength = np.linspace(529.0, 535.0, 181)
    laser = 532.0
    angle = np.pi / 2.0
    temperature, flow, kappa = 300.0, 8.0e4, 3.5
    q = 2.0 * np.sin(angle / 2.0)
    centre = laser * (1.0 - q * flow / 299792458.0)
    sigma = laser * q * np.sqrt(temperature / 510998.95069)
    degrees = 2.0 * kappa - 1.0
    scale = sigma * np.sqrt((degrees - 2.0) / degrees)
    shape = (1.0 + ((wavelength - centre) / scale) ** 2 / degrees) ** (-kappa)
    z = 2.0 * (wavelength - wavelength.min()) / np.ptp(wavelength) - 1.0
    background = 0.04 + 0.006 * z
    measured = 2.3 * shape + background

    result = analyze_thomson_spectral_distribution(
        wavelength,
        measured,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=np.full(wavelength.size, 0.003),
        background_signal=background,
        moment_mc_samples=20,
        physical_model="kappa",
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        physical_fit_options={
            "physical_initial": [280.0, 0.0, 4.0],
            "background_model": "linear",
            "max_nfev": 5000,
        },
    )

    inference = result["physical_inference"]
    assert inference["status"] == "identified_from_measurement"
    assert inference["reportable_as_primary_inference"] is True
    assert inference["gaussian_width_used_as_temperature"] is False
    assert inference["fit"]["primary_result_status"] == "identified_from_measurement"
    assert inference["fit"]["parameter_identifiability"]["joint_jacobian_rank"] == (
        inference["fit"]["parameter_identifiability"]["joint_parameter_count"]
    )
    assert inference["fit"]["electron_temperature_eV"] == pytest.approx(
        temperature, rel=0.05
    )
    assert inference["fit"]["kappa_index"] == pytest.approx(kappa, rel=0.08)



def test_collective_physical_initial_is_routed_to_native_canonical_solver(monkeypatch):
    import tokamak_analysis.thomson_inference as inference

    x, y = _spectrum()
    captured = {}

    def fake_canonical(wavelength, spectra, radius, **kwargs):
        captured.update(kwargs)
        return {
            "channels": [{"electron_temperature_eV": 120.0}],
            "result_classification": {
                "channels": [{
                    "primary": [
                        {"parameter": "electron_density_m3"},
                        {"parameter": "electron_temperature_eV"},
                    ]
                }]
            },
        }

    monkeypatch.setattr(inference, "analyze_thomson_spectra", fake_canonical)
    out = _run(
        x, y,
        physical_model="collective_maxwellian",
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2,
        physical_initial={
            "collective_initial_density_m3": 2.0e19,
            "collective_initial_temperature_eV": 120.0,
        },
        physical_fit_options={"radius_m": 0.2},
    )

    assert captured["collective_initial_density_m3"] == pytest.approx(2.0e19)
    assert captured["collective_initial_temperature_eV"] == pytest.approx(120.0)
    assert out["physical_inference"]["selected_model"] == "collective_maxwellian"
    assert out["physical_inference"]["reportable_as_primary_inference"] is True


def test_collective_kappa_physical_initial_routes_kappa_seed(monkeypatch):
    import tokamak_analysis.thomson_inference as inference

    x, y = _spectrum()
    captured = {}

    def fake_canonical(wavelength, spectra, radius, **kwargs):
        captured.update(kwargs)
        return {
            "channels": [{"electron_temperature_eV": 150.0}],
            "result_classification": {
                "channels": [{
                    "primary": [
                        {"parameter": "electron_density_m3"},
                        {"parameter": "electron_temperature_eV"},
                    ]
                }]
            },
        }

    monkeypatch.setattr(inference, "analyze_thomson_spectra", fake_canonical)
    _run(
        x, y,
        physical_model="collective_kappa",
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2,
        physical_initial={
            "collective_initial_density_m3": 1.5e19,
            "collective_initial_temperature_eV": 150.0,
            "collective_kappa_initial": 3.5,
        },
        physical_fit_options={"radius_m": 0.2},
    )
    assert captured["collective_kappa_initial"] == pytest.approx(3.5)


def test_collective_physical_initial_rejects_unknown_or_duplicate_seed_keys():
    x, y = _spectrum()
    base = {
        "physical_model": "collective_maxwellian",
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": np.pi / 2,
        "physical_fit_options": {"radius_m": 0.2},
    }
    with pytest.raises(ValueError, match="Unsupported physical_initial keys"):
        _run(x, y, **base, physical_initial={"temperature": 100.0})
    with pytest.raises(ValueError, match="not both"):
        _run(
            x, y,
            **{**base, "physical_fit_options": {
                "radius_m": 0.2, "collective_initial_temperature_eV": 100.0
            }},
            physical_initial={"collective_initial_temperature_eV": 100.0},
        )



def test_gaussian_shape_fit_uses_model_and_systematic_covariance_in_likelihood():
    x, y = _spectrum()
    n = x.size
    measurement = np.full(n, 0.015)
    model_covariance = np.eye(n) * 0.025**2
    systematic_covariance = np.eye(n) * 0.010**2
    out = _run(
        x, y,
        signal_uncertainty=measurement,
        model_discrepancy_covariance=model_covariance,
        systematic_covariance=systematic_covariance,
    )
    expected = np.diag(measurement**2) + model_covariance + systematic_covariance
    residual = out["residual"]
    expected_chi_square = float(residual @ np.linalg.solve(expected, residual))
    assert out["chi_square"] == pytest.approx(expected_chi_square, rel=1e-9)
    contract = out["gaussian_fit_likelihood_covariance_components"]
    assert contract["measurement_included"] is True
    assert contract["model_discrepancy_included"] is True
    assert contract["systematic_included"] is True
    assert contract["used_in_optimizer_and_chi_square"] is True


def test_gaussian_shape_fit_rejects_singular_combined_likelihood_covariance():
    x, y = _spectrum()
    zero = np.zeros((x.size, x.size))
    with pytest.raises(ValueError, match="combined measurement, model-discrepancy"):
        _run(
            x, y,
            model_discrepancy_covariance=zero,
            systematic_covariance=zero,
        )



@pytest.mark.parametrize(
    "covariance_name",
    ["model_discrepancy_covariance", "systematic_covariance"],
)
def test_moment_uncertainty_method_reports_discrepancy_and_systematic_only_draws(covariance_name):
    x, y = _spectrum()
    covariance = np.eye(x.size) * 0.25
    result = _run(x, y, moment_mc_samples=40, **{covariance_name: covariance})
    assert result["moment_uncertainty_successful_draws"] >= 20
    assert result["spectral_centroid_uncertainty_nm"] > 0.0
    assert result["moment_uncertainty_method"] == (
        "parametric Monte Carlo from supplied measurement, background, model-discrepancy and systematic covariance components"
    )
    components = result["moment_uncertainty_covariance_components"]
    assert components["model_discrepancy_covariance_included"] is (covariance_name == "model_discrepancy_covariance")
    assert components["systematic_covariance_included"] is (covariance_name == "systematic_covariance")


def test_distribution_routes_scattering_angle_uncertainty_into_projected_evdf():
    x = np.linspace(499.0, 501.0, 81)
    y = 1000.0 * np.exp(-0.5 * ((x - 500.0) / 0.06) ** 2) + 12.0
    result = _run(
        x, y,
        signal_uncertainty=np.full(x.size, 1.0),
        moment_mc_samples=20,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        scattering_angle_uncertainty_rad=2.0e-4,
        projected_evdf_options={
            "regularization_strength": 1.0e-3,
            "velocity_limit_m_s": 8.0e4,
            "velocity_points": 41,
        },
    )
    inverse = result["projected_evdf_inference"]
    assert inverse["scattering_angle_uncertainty_rad"] == pytest.approx(2.0e-4)
    assert inverse["scattering_geometry_covariance_status"] == (
        "first_order_fixed_active_set_implicit_sensitivity"
    )
    assert inverse["evdf_covariance_including_scattering_geometry"] is not None


def test_spectral_distribution_propagates_joint_laser_angle_calibration_into_projected_evdf():
    wavelength = np.linspace(499.8, 500.2, 241)
    laser = 500.0
    angle = np.pi / 2.0
    q = 2.0 * np.sin(angle / 2.0)
    centre = laser * (1.0 - q * 2.0e4 / 299792458.0)
    width = laser * q * 2.5e4 / 299792458.0
    background = np.full_like(wavelength, 3.0)
    signal = 1000.0 * np.exp(-0.5 * ((wavelength - centre) / width) ** 2) + background
    sigma_laser = 2.0e-3
    sigma_angle = 2.0e-4
    cross = 0.25 * sigma_laser * sigma_angle
    result = analyze_thomson_spectral_distribution(
        wavelength, signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        signal_uncertainty=np.full(wavelength.size, 1.0),
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        laser_wavelength_uncertainty_nm=sigma_laser,
        scattering_angle_uncertainty_rad=sigma_angle,
        laser_angle_covariance_nm_rad=cross,
        projected_evdf_options={
            "regularization_strength": 1.0e-3,
            "velocity_limit_m_s": 8.0e4,
            "velocity_points": 81,
        },
    )
    inverse = result["projected_evdf_inference"]
    assert inverse["geometry_uncertainty_covariance_included"] is True
    assert inverse["laser_wavelength_uncertainty_nm"] == pytest.approx(sigma_laser)
    assert inverse["laser_angle_covariance_nm_rad"] == pytest.approx(cross)
    assert inverse["evdf_standard_uncertainty_including_geometry_s_per_m"].shape == (81,)
    assert inverse["velocity_moment_covariance_including_geometry"].shape == (6, 6)


def test_selected_maxwellian_model_executes_canonical_fit_with_top_level_radius(monkeypatch):
    import tokamak_analysis.thomson_inference as inference

    x, y = _spectrum()
    captured = {}

    def fake_canonical(wavelength, spectra, radius, **kwargs):
        captured["wavelength"] = wavelength.copy()
        captured["spectra"] = spectra.copy()
        captured["radius"] = radius.copy()
        captured.update(kwargs)
        return {
            "channels": [{"electron_temperature_eV": 120.0}],
            "result_classification": {
                "channels": [{
                    "primary": [
                        {"parameter": "electron_temperature_eV"},
                        {"parameter": "electron_velocity_projection_m_s"},
                    ]
                }]
            },
        }

    monkeypatch.setattr(inference, "analyze_thomson_spectra", fake_canonical)
    result = _run(
        x, y,
        physical_model="non_collective_maxwellian",
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        radius_m=0.42,
    )

    assert captured["radius"].tolist() == pytest.approx([0.42])
    assert captured["spectra"].shape == (1, x.size)
    assert captured["model"] == "non_collective_maxwellian"
    assert result["physical_inference"]["reportable_as_primary_inference"] is True


def test_relativistic_projected_evdf_retains_rank_deficient_candidate_and_reports_physical_moments():
    """The public product API returns a labelled candidate for rank-deficient data."""
    from tis.products.thomson.api import analyze_spectral_distribution as run_product_analysis

    wavelength = np.linspace(499.7, 500.3, 61)
    background = np.full_like(wavelength, 3.0)
    signal = (
        900.0 * np.exp(-0.5 * ((wavelength - 500.0) / 0.055) ** 2)
        + background
    )
    result = run_product_analysis(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=np.full(wavelength.size, 1.0),
        background_signal=background,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        projected_evdf_options={
            "kinematics": "relativistic_thomson",
            "velocity_limit_m_s": 8.0e4,
            "velocity_points": 101,
            "regularization_strength": 1.0e-3,
        },
    )

    inverse = result["projected_evdf_inference"]
    assert inverse["primary_result_status"] == "not_fully_identifiable_from_measured_spectrum"
    assert inverse["projected_evdf_s_m"] is None
    assert inverse["regularized_candidate_evdf_s_m"] is not None
    assert inverse["moment_candidate_used"] is True
    assert inverse["velocity_parallel_m_s"].shape == (101,)

    moments = inverse["projected_evdf_physical_moments"]
    assert moments["status"] == "conditional_regularized_candidate_not_primary"
    assert moments["primary_result_status"] == "not_fully_identifiable_from_measured_spectrum"
    assert np.isfinite(moments["mean_velocity_parallel_m_s"])
    assert moments["velocity_parallel_variance_m2_s2"] > 0.0
    assert np.isfinite(moments["central_third_velocity_moment_m3_s3"])
    assert moments["central_fourth_velocity_moment_m4_s4"] > 0.0
    assert np.isfinite(moments["skewness"])
    assert np.isfinite(moments["excess_kurtosis"])
    assert moments["moment_covariance"] is None
    assert moments["heat_flux_identifiability"]["status"] == (
        "not_identified_from_single_projected_marginal"
    )
    assert inverse["spectral_moment_closure"]["status"] == (
        "conditional_relativistic_response_aware_moment_closure"
    )
    assert inverse["spectral_moment_closure"]["regularized_candidate_used"] is True


def test_projected_evdf_moments_are_stable_for_narrow_high_flow_distribution():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    mean = 1.0e6
    sigma = 1.0e4
    velocity = np.linspace(mean - 8.0*sigma, mean + 8.0*sigma, 1601)
    pdf = np.exp(-0.5*((velocity-mean)/sigma)**2)
    pdf /= np.trapz(pdf, velocity)
    moments = _projected_evdf_physical_moments(velocity, pdf)

    assert moments["mean_velocity_parallel_m_s"] == pytest.approx(mean, rel=1e-10)
    assert moments["velocity_parallel_variance_m2_s2"] == pytest.approx(sigma**2, rel=2e-4)
    assert moments["central_third_velocity_moment_m3_s3"] == pytest.approx(0.0, abs=1e7)
    assert moments["central_fourth_velocity_moment_m4_s4"] == pytest.approx(
        3.0*sigma**4, rel=3e-4
    )
    assert moments["skewness"] == pytest.approx(0.0, abs=1e-8)
    assert moments["excess_kurtosis"] == pytest.approx(0.0, abs=1e-3)


def test_projected_evdf_physical_moment_covariance_matches_normalized_finite_difference():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.array([-4.0, -2.7, -1.1, -0.2, 0.8, 2.1, 3.9, 6.2]) * 1.0e5
    density = (
        0.72 * np.exp(-0.5 * ((velocity - 0.35e5) / 1.15e5) ** 2)
        + 0.28 * np.exp(-0.5 * ((velocity - 2.1e5) / 0.72e5) ** 2)
    )
    density *= 1.0e-6
    n = velocity.size
    base_covariance = np.fromfunction(
        lambda i, j: 0.35 ** np.abs(i - j), (n, n), dtype=float
    )
    scale = np.sqrt(density)
    covariance = (scale[:, None] * base_covariance) * scale[None, :]
    result = _projected_evdf_physical_moments(velocity, density, covariance)

    keys = (
        "mean_velocity_parallel_m_s",
        "velocity_parallel_variance_m2_s2",
        "velocity_parallel_standard_deviation_m_s",
        "directional_equivalent_temperature_eV",
        "central_third_velocity_moment_m3_s3",
        "central_fourth_velocity_moment_m4_s4",
        "skewness",
        "excess_kurtosis",
    )
    numerical_jacobian = np.empty((len(keys), n))
    for j in range(n):
        step = density[j] * 1.0e-5
        plus = density.copy()
        minus = density.copy()
        plus[j] += step
        minus[j] -= step
        p = _projected_evdf_physical_moments(velocity, plus)
        m = _projected_evdf_physical_moments(velocity, minus)
        numerical_jacobian[:, j] = [
            (p[key] - m[key]) / (2.0 * step) for key in keys
        ]

    expected_covariance = numerical_jacobian @ covariance @ numerical_jacobian.T
    np.testing.assert_allclose(
        result["moment_covariance"],
        expected_covariance,
        rtol=3.0e-5,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result["moment_covariance"],
        result["moment_covariance"].T,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.linalg.eigvalsh(result["moment_covariance"]).min() >= -1.0e-8 * max(
        np.linalg.norm(result["moment_covariance"], ord=2), 1.0
    )


def test_projected_evdf_physical_moment_covariance_rejects_indefinite_input():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-3.0e5, 3.0e5, 9)
    density = np.exp(-0.5 * (velocity / 1.0e5) ** 2)
    covariance = np.eye(velocity.size)
    covariance[0, 1] = covariance[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _projected_evdf_physical_moments(velocity, density, covariance)


def test_thomson_spectral_distribution_executes_through_specialist_product_api():
    from tis.products.thomson.api import analyze_spectral_distribution

    wavelength = np.linspace(499.0, 501.0, 101)
    background = 10.0 + 0.2 * (wavelength - 500.0)
    signal = (
        500.0 * np.exp(-0.5 * ((wavelength - 500.0) / 0.16) ** 2)
        + background
    )
    result = analyze_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
    )
    assert result["spectral_centroid_nm"] == pytest.approx(500.0, abs=0.01)
    assert result["observed_sigma_nm"] == pytest.approx(0.16, rel=0.04)
    assert result["instrument_response_model"] == "measured_response_matrix_forward_operator"


def test_projected_evdf_energy_cdf_matches_one_dimensional_maxwellian():
    from scipy.special import erf
    from scipy.constants import elementary_charge, electron_mass
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    temperature_eV = 80.0
    sigma = np.sqrt(elementary_charge * temperature_eV / electron_mass)
    velocity = np.linspace(-8.0*sigma, 8.0*sigma, 4001)
    density = np.exp(-0.5*(velocity/sigma)**2) / (np.sqrt(2.0*np.pi)*sigma)

    result = _projected_evdf_physical_moments(velocity, density)
    energy = result["projected_parallel_kinetic_energy_cdf_eV"]
    expected = erf(np.sqrt(energy["energy_grid_eV"] / temperature_eV))

    assert energy["distribution_scope"] == "one_dimensional_projected_parallel_energy_only"
    assert energy["cumulative_probability"][0] == pytest.approx(0.0, abs=1e-12)
    assert energy["cumulative_probability"][-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(energy["cumulative_probability"]) >= 0.0)
    np.testing.assert_allclose(
        energy["cumulative_probability"][1:-1],
        expected[1:-1],
        rtol=2.5e-3,
        atol=2.5e-4,
    )



def test_projected_evdf_energy_cdf_covariance_matches_finite_difference():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-4.0e5, 4.0e5, 41)
    density = (
        np.exp(-0.5*((velocity - 2.0e4)/1.15e5)**2)
        + 0.22*np.exp(-0.5*((velocity + 1.8e5)/7.0e4)**2)
    )
    covariance = np.diag((density*2.0e-4)**2)
    result = _projected_evdf_physical_moments(velocity, density, covariance)
    energy = result["projected_parallel_kinetic_energy_cdf_eV"]
    actual = energy["cumulative_probability_covariance"]

    numerical_jacobian = np.empty((velocity.size, velocity.size))
    for column in range(velocity.size):
        step = density[column]*1.0e-5
        plus = density.copy()
        minus = density.copy()
        plus[column] += step
        minus[column] -= step
        cdf_plus = _projected_evdf_physical_moments(
            velocity, plus
        )["projected_parallel_kinetic_energy_cdf_eV"]["cumulative_probability"]
        cdf_minus = _projected_evdf_physical_moments(
            velocity, minus
        )["projected_parallel_kinetic_energy_cdf_eV"]["cumulative_probability"]
        numerical_jacobian[:, column] = (cdf_plus-cdf_minus)/(2.0*step)

    expected = numerical_jacobian @ covariance @ numerical_jacobian.T
    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=1.0e-14)
    np.testing.assert_allclose(actual, actual.T, rtol=1.0e-12, atol=1.0e-14)
    assert np.linalg.eigvalsh(actual).min() >= -1.0e-12
    assert energy["cumulative_probability_standard_uncertainty"][-1] == pytest.approx(0.0, abs=1.0e-12)



def test_projected_evdf_random_energy_moments_match_one_dimensional_maxwellian():
    from scipy.constants import elementary_charge, electron_mass
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    temperature_eV = 80.0
    sigma = np.sqrt(elementary_charge * temperature_eV / electron_mass)
    velocity = np.linspace(-8.0*sigma, 8.0*sigma, 4001)
    density = np.exp(-0.5*(velocity/sigma)**2) / (np.sqrt(2.0*np.pi)*sigma)

    result = _projected_evdf_physical_moments(velocity, density)
    assert result["directional_equivalent_temperature_eV"] == pytest.approx(
        temperature_eV, rel=2.0e-5
    )
    assert result["mean_random_kinetic_energy_parallel_eV"] == pytest.approx(
        temperature_eV / 2.0, rel=2.0e-5
    )
    assert result["random_kinetic_energy_parallel_variance_eV2"] == pytest.approx(
        temperature_eV**2 / 2.0, rel=5.0e-5
    )


def test_projected_evdf_random_energy_moment_covariance_matches_finite_difference():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-4.0e5, 4.0e5, 61)
    density = np.exp(-0.5*((velocity - 2.0e4)/1.15e5)**2)
    covariance = np.diag((density*1.0e-4)**2)
    result = _projected_evdf_physical_moments(velocity, density, covariance)
    names = (
        "mean_random_kinetic_energy_parallel_eV",
        "random_kinetic_energy_parallel_variance_eV2",
    )
    indices = [result["moment_covariance_order"].index(name) for name in names]
    actual = result["moment_covariance"][np.ix_(indices, indices)]

    numerical_jacobian = np.empty((2, velocity.size))
    for column in range(velocity.size):
        step = density[column]*1.0e-5
        plus, minus = density.copy(), density.copy()
        plus[column] += step
        minus[column] -= step
        rp = _projected_evdf_physical_moments(velocity, plus)
        rm = _projected_evdf_physical_moments(velocity, minus)
        for row, name in enumerate(names):
            numerical_jacobian[row, column] = (
                rp[name] - rm[name]
            ) / (2.0*step)

    expected = numerical_jacobian @ covariance @ numerical_jacobian.T
    np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=1.0e-18)
    np.testing.assert_allclose(
        result["random_energy_moment_covariance"], expected,
        rtol=3.0e-4, atol=1.0e-18,
    )
    assert set(result["random_energy_moment_standard_uncertainty"]) == set(names)

def test_projected_evdf_parallel_heat_flux_matches_analytic_skewed_mixture():
    """A skewed 1-D EVDF has nonzero q_parallel^(1D) = m_e*n*mu3/2."""
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-5.0, 5.0, 501)
    sigma = 0.5
    # A 3:1 mixture with component means -0.5 and +1.5 has zero
    # mixture mean and analytic third central moment mu3 = 0.75 (m/s)^3.
    pdf = (
        0.75 * np.exp(-0.5 * ((velocity + 0.5) / sigma) ** 2)
        + 0.25 * np.exp(-0.5 * ((velocity - 1.5) / sigma) ** 2)
    ) / (np.sqrt(2.0 * np.pi) * sigma)
    density = 2.0e19
    density_sigma = 1.0e18
    covariance = np.diag(np.full(velocity.size, 1.0e-14))
    result = _projected_evdf_physical_moments(
        velocity, pdf, covariance,
        electron_density_m3=density,
        electron_density_uncertainty_m3=density_sigma,
    )

    me = 9.1093837139e-31
    expected_mu3 = 0.75
    expected_flux = 0.5 * me * density * expected_mu3
    assert result["central_third_velocity_moment_m3_s3"] == pytest.approx(
        expected_mu3, rel=2e-3
    )
    assert result["parallel_projected_heat_flux_W_m2"] == pytest.approx(
        expected_flux, rel=2e-3
    )
    assert result["parallel_projected_heat_flux_per_density_W_m"] == pytest.approx(
        0.5 * me * expected_mu3, rel=2e-3
    )
    mu3_index = result["moment_covariance_order"].index(
        "central_third_velocity_moment_m3_s3"
    )
    sigma_mu3 = np.sqrt(result["moment_covariance"][mu3_index, mu3_index])
    expected_flux_sigma = np.sqrt(
        (0.5 * me * density * sigma_mu3) ** 2
        + (0.5 * me * expected_mu3 * density_sigma) ** 2
    )
    assert result["parallel_projected_heat_flux_standard_uncertainty_W_m2"] == pytest.approx(
        expected_flux_sigma, rel=1e-10
    )
    assert "not_the_full_three_dimensional" in result["heat_flux_physical_scope"]
    assert "parallel_projected_heat_flux" in result["equations"]

    from tis.products.thomson.physics.thomson_distribution import (
        infer_projected_parallel_heat_flux,
    )
    formula_result = infer_projected_parallel_heat_flux(
        velocity, pdf, covariance,
        electron_density_m3=density,
        electron_density_uncertainty_m3=density_sigma,
    )
    assert formula_result["parallel_projected_heat_flux_W_m2"] == pytest.approx(
        expected_flux, rel=2e-3
    )
    assert set(formula_result) == {
        "parallel_projected_heat_flux_per_density_W_m",
        "parallel_projected_heat_flux_W_m2",
        "parallel_projected_heat_flux_standard_uncertainty_W_m2",
        "heat_flux_physical_scope",
    }


def test_projected_evdf_heat_flux_does_not_invent_density_or_three_dimensional_flux():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-5.0, 5.0, 501)
    pdf = np.exp(-0.5 * velocity**2) / np.sqrt(2.0 * np.pi)
    result = _projected_evdf_physical_moments(velocity, pdf)
    assert result["parallel_projected_heat_flux_W_m2"] is None
    assert result["parallel_projected_heat_flux_per_density_W_m"] == pytest.approx(0.0, abs=1e-30)
    assert result["heat_flux_physical_scope"].startswith("one_dimensional_parallel")
    with pytest.raises(ValueError, match="electron_density_m3"):
        _projected_evdf_physical_moments(
            velocity, pdf, electron_density_m3="invented"
        )
    with pytest.raises(ValueError, match="requires an independently supplied"):
        _projected_evdf_physical_moments(
            velocity, pdf, electron_density_uncertainty_m3=1.0e17
        )


def test_projected_evdf_heat_flux_is_exposed_through_product_api():
    from tis.products.thomson.api import analyze_spectral_distribution

    wavelength = np.linspace(520.0, 544.0, 241)
    background = np.full_like(wavelength, 0.02)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 1.0) ** 2) + background
    result = analyze_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        signal_uncertainty=np.full_like(wavelength, 1e-3),
        projected_evdf_options={
            "velocity_limit_m_s": 1.0e6,
            "velocity_points": 81,
            "regularization_strength": 1e-3,
            "electron_density_m3": 2.0e19,
            "electron_density_uncertainty_m3": 1.0e18,
            "electron_density_mu3_cross_covariance_s_minus3": 0.0,
        },
    )
    moments = result["projected_evdf_inference"]["projected_evdf_physical_moments"]
    assert moments["heat_flux_density_input_m3"] == pytest.approx(2.0e19)
    assert moments["heat_flux_density_mu3_cross_covariance_input_s_minus3"] == pytest.approx(0.0)
    assert np.isfinite(moments["parallel_projected_heat_flux_per_density_W_m"])
    assert np.isfinite(moments["parallel_projected_heat_flux_W_m2"])
    assert moments["heat_flux_physical_scope"].startswith("one_dimensional_parallel")


def test_projected_heat_flux_propagates_density_evdf_cross_covariance():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-4.0e4, 4.0e4, 9)
    pdf = np.array([0.02, 0.05, 0.10, 0.18, 0.24, 0.19, 0.12, 0.07, 0.03])
    pdf /= np.trapz(pdf, velocity)
    evdf_covariance = np.eye(velocity.size) * 1.0e-8
    ne = 2.0e19
    dne = 1.0e17

    independent = _projected_evdf_physical_moments(
        velocity, pdf, evdf_covariance,
        electron_density_m3=ne,
        electron_density_uncertainty_m3=dne,
    )
    mu3_index = independent["moment_covariance_order"].index(
        "central_third_velocity_moment_m3_s3"
    )
    var_mu3 = independent["moment_covariance"][mu3_index, mu3_index]
    cov_ne_mu3 = 0.35 * np.sqrt(dne**2 * var_mu3)
    correlated = _projected_evdf_physical_moments(
        velocity, pdf, evdf_covariance,
        electron_density_m3=ne,
        electron_density_uncertainty_m3=dne,
        electron_density_mu3_cross_covariance_s_minus3=cov_ne_mu3,
    )
    mu3 = correlated["central_third_velocity_moment_m3_s3"]
    expected_variance = (0.5 * 9.1093837139e-31)**2 * (
        ne**2 * var_mu3 + mu3**2 * dne**2 + 2.0 * ne * mu3 * cov_ne_mu3
    )
    assert correlated["parallel_projected_heat_flux_standard_uncertainty_W_m2"] == pytest.approx(
        np.sqrt(expected_variance), rel=1e-12
    )
    assert correlated["heat_flux_density_mu3_cross_covariance_input_s_minus3"] == pytest.approx(
        cov_ne_mu3
    )
    assert correlated["parallel_projected_heat_flux_standard_uncertainty_W_m2"] != pytest.approx(
        independent["parallel_projected_heat_flux_standard_uncertainty_W_m2"]
    )
    from tis.products.thomson.physics.thomson_distribution import (
        infer_projected_parallel_heat_flux,
    )
    helper_result = infer_projected_parallel_heat_flux(
        velocity, pdf, evdf_covariance,
        electron_density_m3=ne,
        electron_density_uncertainty_m3=dne,
        electron_density_mu3_cross_covariance_s_minus3=cov_ne_mu3,
    )
    assert helper_result["parallel_projected_heat_flux_standard_uncertainty_W_m2"] == pytest.approx(
        correlated["parallel_projected_heat_flux_standard_uncertainty_W_m2"],
        rel=1e-12,
    )


def test_projected_heat_flux_rejects_non_psd_density_evdf_cross_covariance():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-3.0e4, 3.0e4, 9)
    pdf = np.exp(-0.5 * (velocity / 1.0e4) ** 2)
    covariance = np.eye(velocity.size) * 1.0e-10
    with pytest.raises(ValueError, match="joint density/EVDF covariance PSD bound"):
        _projected_evdf_physical_moments(
            velocity, pdf, covariance,
            electron_density_m3=1.0e19,
            electron_density_uncertainty_m3=1.0e16,
            electron_density_mu3_cross_covariance_s_minus3=1.0e100,
        )


def test_correlated_projected_heat_flux_relation_is_registered_in_both_formula_catalogues():
    from pathlib import Path

    name = "Correlated density and projected-EVDF heat-flux uncertainty"
    root_registry = json.loads(Path("FORMULA_REGISTRY.json").read_text())
    packaged_registry = json.loads(Path("tis/data/FORMULA_REGISTRY.json").read_text())
    root_entry = next(item for item in root_registry if item["name"] == name)
    packaged_entry = next(item for item in packaged_registry if item["name"] == name)

    assert root_entry == packaged_entry
    assert "2 n_e mu_3 Cov(n_e,mu_3)" in root_entry["equation"]
    contract = root_entry["execution_contract"]
    assert contract["implementation_link_status"] == "mapped"
    assert contract["function_link"].endswith(".infer_projected_parallel_heat_flux")
    assert "electron_density_mu3_cross_covariance_s_minus3" in {
        item["name"] for item in contract["input_specs"]
    }
    assert all(Path(link.split("::")[0]).exists() for link in contract["test_links"])


def test_projected_heat_flux_cross_covariance_is_in_dashboard_diagnostic_contract():
    from pathlib import Path

    name = "Thomson spectral distribution"
    root = json.loads(Path("DIAGNOSTIC_REGISTRY.json").read_text())
    packaged = json.loads(Path("tis/data/DIAGNOSTIC_REGISTRY.json").read_text())
    root_contract = next(row for row in root if row["diagnostic"] == name)["execution_contract"]
    packaged_contract = next(row for row in packaged if row["diagnostic"] == name)["execution_contract"]

    assert root_contract == packaged_contract
    options = next(
        item for item in root_contract["input_specs"]
        if item["name"] == "projected_evdf_options"
    )
    assert "electron_density_mu3_cross_covariance_s_minus3" in options["description"]
    assert "heat_flux_density_mu3_cross_covariance_input_s_minus3" in {
        item["name"] for item in root_contract["output_specs"]
    }
    assert any(
        "test_projected_heat_flux_propagates_density_evdf_cross_covariance" in item
        for item in root_contract["test_links"]
    )



def test_piecewise_linear_evdf_polynomial_moments_are_cell_exact_on_nonuniform_grid():
    from tis.products.thomson.physics.thomson_distribution import (
        _piecewise_linear_moment_weights,
    )

    velocity = np.array([-3.0, -1.7, -0.2, 0.4, 2.8, 5.1])
    # A piecewise-linear density with deliberately nonuniform nodes.
    density = np.array([0.4, 1.2, 0.7, 1.8, 0.5, 0.9])
    center = 0.37
    nodes, weights = np.polynomial.legendre.leggauss(12)
    for power in range(5):
        coefficients = _piecewise_linear_moment_weights(
            velocity, power, center
        )
        actual = float(coefficients @ density)
        expected = 0.0
        for index, (left, right) in enumerate(zip(velocity[:-1], velocity[1:])):
            half_width = 0.5 * (right - left)
            midpoint = 0.5 * (right + left)
            points = midpoint + half_width * nodes
            fraction = (points - left) / (right - left)
            interpolated = (
                (1.0 - fraction) * density[index]
                + fraction * density[index + 1]
            )
            expected += half_width * float(
                weights @ (interpolated * (points - center) ** power)
            )
        assert actual == pytest.approx(expected, rel=2e-13, abs=2e-13)

    # Zeroth-order integration must remain identical to the inverse solver's
    # composite trapezoidal normalization contract.
    zeroth = _piecewise_linear_moment_weights(velocity, 0)
    assert float(zeroth @ density) == pytest.approx(
        float(np.trapz(density, velocity)), rel=2e-14, abs=2e-14
    )


def test_piecewise_linear_evdf_moment_covariance_uses_exact_cell_integrals():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.array([-5.0, -3.2, -1.1, -0.3, 1.4, 4.7, 7.0])
    density = (
        0.65 * np.exp(-0.5 * ((velocity + 0.4) / 1.5) ** 2)
        + 0.35 * np.exp(-0.5 * ((velocity - 2.3) / 0.9) ** 2)
    )
    covariance = np.diag((0.015 * density + 1e-5) ** 2)
    result = _projected_evdf_physical_moments(velocity, density, covariance)
    keys = (
        "mean_velocity_parallel_m_s",
        "velocity_parallel_variance_m2_s2",
        "central_third_velocity_moment_m3_s3",
        "central_fourth_velocity_moment_m4_s4",
        "skewness",
        "excess_kurtosis",
    )
    numerical = np.empty((len(keys), velocity.size))
    for column in range(velocity.size):
        step = density[column] * 1e-5
        plus, minus = density.copy(), density.copy()
        plus[column] += step
        minus[column] -= step
        p = _projected_evdf_physical_moments(velocity, plus)
        m = _projected_evdf_physical_moments(velocity, minus)
        numerical[:, column] = [
            (p[key] - m[key]) / (2.0 * step) for key in keys
        ]
    expected = numerical @ covariance @ numerical.T
    indices = [result["moment_covariance_order"].index(key) for key in keys]
    actual = result["moment_covariance"][np.ix_(indices, indices)]
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=1e-12)



def test_analysis_81_contract_exposes_the_physical_evdf_moment_covariance():
    from tokamak_analysis.analysis_registry import get_analysis

    entry = get_analysis(81)
    contract = entry["execution_contract"]
    inputs = {item["name"]: item for item in contract["input_specs"]}
    outputs = {item["name"]: item for item in contract["output_specs"]}

    assert "projected_evdf_options" in inputs
    assert "electron_density_mu3_cross_covariance_s_minus3" in inputs[
        "projected_evdf_options"
    ]["description"]
    required_outputs = {
        "central_third_velocity_moment_m3_s3",
        "central_fourth_velocity_moment_m4_s4",
        "skewness",
        "excess_kurtosis",
        "moment_covariance",
        "parallel_projected_pressure_heat_flux_joint_covariance",
        "density_evdf_moment_joint_covariance",
        "parallel_projected_heat_flux_W_m2",
        "parallel_projected_pressure_Pa",
    }
    assert required_outputs.issubset(outputs)
    assert "not a 3-D pressure tensor or heat flux" in inputs[
        "projected_evdf_options"
    ]["description"]



def test_projected_evdf_moments_recover_analytic_two_population_velocity_moments_and_flux():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-8.0e5, 8.0e5, 4001)
    fractions = np.array([0.65, 0.35])
    means = np.array([-0.55e5, 1.10e5])
    sigmas = np.array([0.72e5, 0.43e5])
    pdf = sum(
        fraction * np.exp(-0.5 * ((velocity - mean) / sigma) ** 2)
        / (np.sqrt(2.0 * np.pi) * sigma)
        for fraction, mean, sigma in zip(fractions, means, sigmas)
    )
    pdf /= np.trapz(pdf, velocity)

    expected_mean = float(fractions @ means)
    offsets = means - expected_mean
    component_variances = sigmas**2
    expected_variance = float(
        fractions @ (component_variances + offsets**2)
    )
    expected_mu3 = float(
        fractions @ (offsets**3 + 3.0 * offsets * component_variances)
    )
    expected_mu4 = float(
        fractions @ (
            offsets**4 + 6.0 * offsets**2 * component_variances
            + 3.0 * component_variances**2
        )
    )
    density = 4.0e18
    result = _projected_evdf_physical_moments(
        velocity, pdf, electron_density_m3=density,
    )

    electron_mass = 9.1093837139e-31
    elementary_charge = 1.602176634e-19
    assert result['mean_velocity_parallel_m_s'] == pytest.approx(expected_mean, rel=2e-5)
    assert result['velocity_parallel_variance_m2_s2'] == pytest.approx(expected_variance, rel=2e-5)
    assert result['central_third_velocity_moment_m3_s3'] == pytest.approx(expected_mu3, rel=2e-4)
    assert result['central_fourth_velocity_moment_m4_s4'] == pytest.approx(expected_mu4, rel=2e-4)
    assert result['directional_equivalent_temperature_eV'] == pytest.approx(
        electron_mass * expected_variance / elementary_charge, rel=2e-5
    )
    assert result['parallel_projected_pressure_Pa'] == pytest.approx(
        electron_mass * density * expected_variance, rel=2e-5
    )
    assert result['parallel_projected_heat_flux_W_m2'] == pytest.approx(
        0.5 * electron_mass * density * expected_mu3, rel=2e-4
    )
    assert result['heat_flux_physical_scope'].startswith('one_dimensional_parallel_kinetic_energy_flux_component_only')
    assert result['heat_flux_identifiability']['status'] == 'not_identified_from_single_projected_marginal'


def test_projected_evdf_joint_pressure_flux_covariance_is_psd_and_density_is_explicit():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_physical_moments,
    )

    velocity = np.linspace(-5.0e5, 5.0e5, 301)
    pdf = np.exp(-0.5 * (velocity / 1.2e5) ** 2) * (
        1.0 + 0.25 * np.tanh(velocity / 1.5e5)
    )
    pdf /= np.trapz(pdf, velocity)
    evdf_covariance = np.diag((pdf * 0.01) ** 2)
    result = _projected_evdf_physical_moments(
        velocity, pdf, evdf_covariance,
        electron_density_m3=3.0e18,
        electron_density_uncertainty_m3=1.5e17,
    )

    joint = result['parallel_pressure_heat_flux_joint_covariance']
    assert joint.shape == (2, 2)
    assert np.linalg.eigvalsh(joint).min() >= -1e-10 * max(np.max(np.abs(joint)), 1e-300)
    assert result['parallel_pressure_heat_flux_joint_standard_uncertainty'].shape == (2,)
    assert result['density_evdf_moment_joint_covariance'].shape == (3, 3)
    assert result['parallel_pressure_density_input_m3'] == pytest.approx(3.0e18)


def test_registered_product_api_executes_spectral_distribution_and_returns_physical_evdf_moments():
    from tis.products.thomson.api import analyze_spectral_distribution

    wavelength = np.linspace(520.0, 544.0, 241)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 1.0) ** 2) + 0.02
    result = analyze_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.full_like(wavelength, 0.02),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        signal_uncertainty=np.full_like(wavelength, 1e-3),
        projected_evdf_options={
            'velocity_limit_m_s': 1.0e6,
            'velocity_points': 81,
            'regularization_strength': None,
        },
    )
    inverse = result['projected_evdf_inference']
    moments = inverse['projected_evdf_physical_moments']
    assert inverse['source_spectrum'] == 'same_calibrated_measured_spectrum_as_spectral_distribution'
    assert inverse['inference_scope'] == 'one_dimensional_q_parallel_velocity_marginal_only'
    assert moments['velocity_projection'] == 'q_parallel_one_dimensional_marginal'
    assert moments['velocity_parallel_variance_m2_s2'] > 0.0
    assert np.isfinite(moments['central_third_velocity_moment_m3_s3'])
    assert np.isfinite(moments['central_fourth_velocity_moment_m4_s4'])
    assert moments['moment_covariance'] is not None
    assert inverse['spectral_moment_closure']['comparison_performed'] is True



def test_relativistic_projected_moments_integrate_piecewise_linear_evdf_not_nodal_trapezoids():
    from scipy.integrate import quad
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_relativistic_lab_moments,
    )

    c = 299792458.0
    me = 9.1093837139e-31
    e = 1.602176634e-19
    velocity = c * np.array([-0.72, -0.51, -0.23, 0.04, 0.31, 0.58, 0.79])
    density = np.array([0.03, 0.18, 0.62, 1.0, 0.74, 0.29, 0.06])
    result = _projected_evdf_relativistic_lab_moments(velocity, density)
    norm = np.trapz(density, velocity)

    def f(v):
        return float(np.interp(v, velocity, density))

    def energy(v):
        return (1.0 / np.sqrt(1.0 - (v / c) ** 2) - 1.0) * me * c**2

    expected_mean = quad(lambda v: v * f(v), velocity[0], velocity[-1], points=velocity[1:-1], epsabs=1e-5)[0] / norm
    expected_energy = quad(lambda v: energy(v) * f(v), velocity[0], velocity[-1], points=velocity[1:-1], epsabs=1e-12)[0] / norm
    expected_flux = quad(lambda v: energy(v) * v * f(v), velocity[0], velocity[-1], points=velocity[1:-1], epsabs=1e-5)[0] / norm

    assert result["integration_method"] == "piecewise_linear_evdf_interval_gauss_legendre_24_point"
    assert result["mean_velocity_parallel_m_s"] == pytest.approx(expected_mean, rel=2e-10)
    assert result["mean_kinetic_energy_eV"] == pytest.approx(expected_energy / e, rel=2e-10)
    assert result["kinetic_energy_flux_per_density_W_m"] == pytest.approx(expected_flux, rel=2e-10)


def test_relativistic_projected_moment_covariance_matches_normalized_functional_finite_difference():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_relativistic_lab_moments,
    )

    c = 299792458.0
    velocity = c * np.array([-0.32, -0.21, -0.08, 0.03, 0.17, 0.29])
    density = np.array([0.08, 0.31, 0.88, 1.0, 0.56, 0.12])
    factor = np.diag(np.array([1e-6, 2e-6, 3e-6, 2e-6, 1e-6, 1e-6]))
    covariance = factor @ factor.T
    result = _projected_evdf_relativistic_lab_moments(
        velocity, density, covariance
    )
    predicted = result["moment_covariance"]
    numerical_jacobian = np.zeros((3, velocity.size))
    for index in range(velocity.size):
        step = 1e-5 * max(abs(density[index]), 0.1)
        plus = density.copy()
        minus = density.copy()
        plus[index] += step
        minus[index] -= step
        if minus[index] < 0.0:
            minus[index] = density[index]
            derivative_denominator = step
            rp = _projected_evdf_relativistic_lab_moments(velocity, plus)
            rm = _projected_evdf_relativistic_lab_moments(velocity, minus)
            vp = np.array([
                rp["mean_velocity_parallel_m_s"],
                rp["mean_kinetic_energy_eV"] * 1.602176634e-19,
                rp["kinetic_energy_flux_per_density_W_m"],
            ])
            vm = np.array([
                rm["mean_velocity_parallel_m_s"],
                rm["mean_kinetic_energy_eV"] * 1.602176634e-19,
                rm["kinetic_energy_flux_per_density_W_m"],
            ])
            numerical_jacobian[:, index] = (vp - vm) / derivative_denominator
        else:
            rp = _projected_evdf_relativistic_lab_moments(velocity, plus)
            rm = _projected_evdf_relativistic_lab_moments(velocity, minus)
            vp = np.array([
                rp["mean_velocity_parallel_m_s"],
                rp["mean_kinetic_energy_eV"] * 1.602176634e-19,
                rp["kinetic_energy_flux_per_density_W_m"],
            ])
            vm = np.array([
                rm["mean_velocity_parallel_m_s"],
                rm["mean_kinetic_energy_eV"] * 1.602176634e-19,
                rm["kinetic_energy_flux_per_density_W_m"],
            ])
            numerical_jacobian[:, index] = (vp - vm) / (2.0 * step)
    numerical_covariance = numerical_jacobian @ covariance @ numerical_jacobian.T
    assert predicted == pytest.approx(numerical_covariance, rel=2e-5, abs=1e-10)
    assert np.linalg.eigvalsh(predicted).min() >= -1e-12
    uncertainty = result["moment_standard_uncertainty"]
    assert uncertainty["mean_kinetic_energy_eV"] == pytest.approx(
        uncertainty["mean_kinetic_energy_J"] / 1.602176634e-19
    )
    assert uncertainty["mean_lorentz_factor"] == pytest.approx(
        uncertainty["mean_kinetic_energy_J"] / (9.1093837139e-31 * c**2)
    )



def test_relativistic_projected_lab_energy_flux_uses_only_explicit_density_and_joint_uncertainty():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_relativistic_lab_moments,
    )

    c = 299792458.0
    velocity = c * np.array([-0.18, -0.09, 0.02, 0.11, 0.2])
    density = np.array([0.08, 0.55, 1.0, 0.62, 0.12])
    factor = np.diag([2e-5, 3e-5, 2e-5, 3e-5, 2e-5])
    evdf_covariance = factor @ factor.T
    without_density = _projected_evdf_relativistic_lab_moments(
        velocity, density, evdf_covariance
    )
    assert without_density["kinetic_energy_flux_W_m2"] is None

    ne = 2.4e19
    dne = 1.2e18
    independent = _projected_evdf_relativistic_lab_moments(
        velocity, density, evdf_covariance,
        electron_density_m3=ne,
        electron_density_uncertainty_m3=dne,
    )
    flux_per_density = independent["kinetic_energy_flux_per_density_W_m"]
    variance_per_density = independent["moment_covariance"][2, 2]
    expected_flux = ne * flux_per_density
    expected_variance = ne**2 * variance_per_density + flux_per_density**2 * dne**2
    assert independent["kinetic_energy_flux_W_m2"] == pytest.approx(expected_flux)
    assert independent["kinetic_energy_flux_standard_uncertainty_W_m2"] == pytest.approx(
        np.sqrt(expected_variance)
    )

    cross = 0.25 * np.sqrt(dne**2 * variance_per_density)
    correlated = _projected_evdf_relativistic_lab_moments(
        velocity, density, evdf_covariance,
        electron_density_m3=ne,
        electron_density_uncertainty_m3=dne,
        electron_density_flux_per_density_cross_covariance_W_m2=cross,
    )
    correlated_variance = (
        ne**2 * variance_per_density
        + flux_per_density**2 * dne**2
        + 2.0 * ne * flux_per_density * cross
    )
    assert correlated["kinetic_energy_flux_standard_uncertainty_W_m2"] == pytest.approx(
        np.sqrt(correlated_variance)
    )
    assert correlated["kinetic_energy_flux_uncertainty_assumption"] == (
        "joint_density_evdf_flux_covariance"
    )


def test_relativistic_projected_lab_flux_rejects_density_cross_covariance_without_joint_inputs():
    from tis.products.thomson.physics.thomson_distribution import (
        _projected_evdf_relativistic_lab_moments,
    )

    c = 299792458.0
    velocity = c * np.linspace(-0.2, 0.2, 5)
    density = np.array([0.1, 0.6, 1.0, 0.6, 0.1])
    with pytest.raises(ValueError, match="requires EVDF covariance"):
        _projected_evdf_relativistic_lab_moments(
            velocity, density,
            electron_density_m3=1e19,
            electron_density_flux_per_density_cross_covariance_W_m2=0.0,
        )



def test_analysis_81_registry_and_formula_registry_expose_relativistic_lab_flux_contract():
    from pathlib import Path
    from tokamak_analysis.analysis_registry import get_analysis

    entry = get_analysis(81)
    contract = entry["execution_contract"]
    projected_spec = next(
        item for item in contract["input_specs"]
        if item["name"] == "projected_evdf_options"
    )
    assert "electron_density_flux_per_density_cross_covariance_W_m2" in projected_spec["description"]
    output_names = {item["name"] for item in contract["output_specs"]}
    assert "relativistic_lab_frame_moments" in output_names
    assert "kinetic_energy_flux_W_m2" in output_names
    assert "kinetic_energy_flux_standard_uncertainty_W_m2" in output_names

    formula_registry = json.loads(
        (Path(__file__).resolve().parents[1] / "FORMULA_REGISTRY.json").read_text(
            encoding="utf-8"
        )
    )
    formula = next(
        item for item in formula_registry
        if item.get("execution_contract", {}).get("formula_key")
        == "thomson_relativistic_projected_lab_energy_flux"
    )
    assert formula["execution_contract"]["implementation_link_status"] == "mapped"
    assert formula["execution_contract"]["test_links"]
