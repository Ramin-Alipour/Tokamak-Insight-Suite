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
    closure = inverse["spectral_moment_closure"]
    assert closure["comparison_performed"] is True
    assert closure["status"] == "conditional_moment_closure"
    assert np.isfinite(closure["centroid_closure_residual_nm"])
    assert np.isfinite(closure["variance_closure_residual_nm2"])


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
