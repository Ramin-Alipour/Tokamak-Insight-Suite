import numpy as np
import pytest

from tokamak_analysis.thomson_evdf import reconstruct_thomson_evdf


def _case():
    wavelength = np.linspace(499.8, 500.2, 401)
    laser = 500.0
    theta = np.pi / 2
    velocity = 2.0e4
    sigma_v = 2.5e4
    q_over_ki = 2 * np.sin(theta / 2)
    centre = laser * (1 - q_over_ki * velocity / 299_792_458.0)
    sigma_lambda = laser * q_over_ki * sigma_v / 299_792_458.0
    background = np.full(wavelength.size, 3.0)
    signal = 1000 * np.exp(-0.5 * ((wavelength - centre) / sigma_lambda) ** 2) + background
    return wavelength, signal, background


def _run(wavelength, signal, background, **kwargs):
    return reconstruct_thomson_evdf(
        wavelength, signal,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        regularization_strength=1e-3,
        velocity_limit_m_s=8.0e4,
        velocity_points=161,
        **kwargs,
    )


def test_evdf_trapezoidal_normalization_jacobian_matches_finite_difference():
    from tokamak_analysis.thomson_evdf import _normalize_evdf_coefficients

    velocity = np.linspace(-3.0, 4.0, 9)
    coefficients = np.array([0.4, 1.1, 0.7, 2.0, 1.3, 0.8, 1.6, 0.9, 0.5])
    distribution, analytic_jacobian = _normalize_evdf_coefficients(coefficients, velocity)
    assert np.trapz(distribution, velocity) == pytest.approx(1.0, abs=1e-13)
    numerical_jacobian = np.empty_like(analytic_jacobian)
    step = 1e-6
    for index in range(coefficients.size):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += step
        minus[index] -= step
        f_plus, _ = _normalize_evdf_coefficients(plus, velocity)
        f_minus, _ = _normalize_evdf_coefficients(minus, velocity)
        numerical_jacobian[:, index] = (f_plus - f_minus) / (2.0 * step)
    np.testing.assert_allclose(analytic_jacobian, numerical_jacobian, rtol=2e-7, atol=2e-9)


def test_evdf_reconstruction_returns_normalized_nonnegative_1d_projection():
    wavelength, signal, background = _case()
    out = _run(wavelength, signal, background)
    f = out["evdf_parallel_s_per_m"]
    v = out["velocity_parallel_m_s"]
    assert np.all(f >= 0)
    assert np.trapz(f, v) == pytest.approx(1.0, rel=2e-5)
    assert out["scattering_vector_projection_only"] is True
    assert out["absolute_density_identified"] is False
    assert out["directional_equivalent_temperature_eV"] > 0
    assert out["fitted_spectrum"].shape == signal.shape
    assert set(out["parametric_model_comparison"]) == {"maxwellian", "kappa_student_t", "super_gaussian"}
    assert set(out["parametric_models"]) == {"maxwellian", "kappa_student_t", "super_gaussian"}
    assert all("aic" in item and "bic" in item for item in out["parametric_model_comparison"].values())
    # Without supplied noise, the residual variance is estimated and the
    # conventional Gaussian information-criterion form includes n*log(RSS/n).
    for item in out["parametric_model_comparison"].values():
        n = signal.size
        k = len(item["parameters"])
        rss = item["weighted_residual_sum_squares"]
        expected = n * np.log(max(rss, np.finfo(float).tiny) / n)
        assert item["information_criterion_likelihood"] == "unknown_common_residual_variance_estimated_from_unweighted_residuals"
        assert item["aic"] == pytest.approx(expected + 2 * k)
        assert item["bic"] == pytest.approx(expected + k * np.log(n))


def test_evdf_selects_regularization_by_gcv_and_reports_conditional_covariance():
    wavelength, signal, background = _case()
    out = reconstruct_thomson_evdf(
        wavelength, signal,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        regularization_strength=None,
        velocity_limit_m_s=8.0e4,
        velocity_points=81,
        signal_uncertainty=np.full(signal.size, 1.0),
    )
    assert out["regularization_selection"] == "generalized_cross_validation_over_logarithmic_candidate_grid"
    assert np.isfinite(out["regularization_strength"]) and out["regularization_strength"] > 0
    assert out["regularization_gcv_scores"].shape == out["regularization_candidates"].shape
    assert out["evdf_covariance_available"] is True
    assert out["evdf_covariance"].shape == (81, 81)
    assert out["evdf_standard_uncertainty_s_per_m"].shape == (81,)
    assert "conditional on the selected grid" in out["evdf_uncertainty_interpretation"]


def test_evdf_parametric_comparison_reports_conditional_parameter_uncertainty():
    wavelength, signal, background = _case()
    out = _run(
        wavelength, signal, background,
        signal_uncertainty=np.full(signal.size, 1.0),
    )
    fit = out["parametric_model_comparison"]["maxwellian"]
    assert fit["success"] is True
    assert fit["parameter_covariance"].shape == (3, 3)
    assert fit["parameter_standard_uncertainty"].shape == (3,)
    assert np.all(np.isfinite(fit["parameter_standard_uncertainty"]))
    assert fit["parameter_uncertainty_status"] in {
        "conditional_local_gaussian_covariance",
        "boundary_limited_for_some_parameters",
    }
    assert fit["parameter_covariance_interpretation"].startswith("Local linearized covariance")
    assert fit["parameter_one_sigma_intervals"]["mean_velocity_parallel_m_s"] is not None
    # Supplied measurement uncertainty defines a known Gaussian likelihood:
    # AIC/BIC use chi-square, not the unknown-variance n*log(RSS/n) form.
    n = signal.size
    k = len(fit["parameters"])
    rss = fit["weighted_residual_sum_squares"]
    assert fit["information_criterion_likelihood"] == "known_measurement_covariance_gaussian_likelihood_up_to_parameter_independent_constant"
    assert fit["aic"] == pytest.approx(rss + 2 * k)
    assert fit["bic"] == pytest.approx(rss + k * np.log(n))


def test_evdf_requires_explicit_positive_regularization_and_response():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="regularization_strength"):
        reconstruct_thomson_evdf(
            wavelength, signal, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=background, regularization_strength=0.0,
        )
    with pytest.raises(ValueError, match="positive integrated response"):
        reconstruct_thomson_evdf(
            wavelength, signal, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2,
            instrument_response_matrix=np.zeros((wavelength.size, wavelength.size)),
            background_signal=background, regularization_strength=1e-3,
        )


def test_evdf_rejects_incompatible_uncertainty_inputs():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="not both"):
        _run(
            wavelength, signal, background,
            signal_uncertainty=np.ones(signal.size),
            signal_covariance=np.eye(signal.size),
        )


def test_evdf_does_not_claim_a_3d_distribution_or_scalar_temperature():
    wavelength, signal, background = _case()
    out = _run(wavelength, signal, background)
    assert "unrestricted three-dimensional EVDF from one scattering geometry" in out["identifiability"]["non_identifiable"]
    assert "directional_equivalent_temperature_eV" in out
    assert "electron_temperature_eV" not in out


def test_evdf_parametric_comparison_does_not_score_nonconverged_optimizer(monkeypatch):
    from types import SimpleNamespace
    import tokamak_analysis.thomson_evdf as evdf

    wavelength, signal, background = _case()

    def failed_optimizer(_objective, initial, **_kwargs):
        return SimpleNamespace(
            success=False,
            x=np.asarray(initial, dtype=float),
            message="forced non-convergence for regression test",
        )

    monkeypatch.setattr(evdf, "least_squares", failed_optimizer)
    out = _run(wavelength, signal, background)
    for fit in out["parametric_model_comparison"].values():
        assert fit["success"] is False
        assert fit["weighted_residual_sum_squares"] is None
        assert fit["aic"] is None
        assert fit["bic"] is None
        assert fit["parameter_uncertainty_status"] == "unavailable_optimizer_not_converged"


def test_evdf_rejects_non_normalized_measured_response_without_renormalizing():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="already sum to one"):
        reconstruct_thomson_evdf(
            wavelength, signal,
            laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2,
            instrument_response_matrix=2.0 * np.eye(wavelength.size),
            background_signal=background,
            regularization_strength=1e-3,
            velocity_limit_m_s=8.0e4,
            velocity_points=81,
        )


def test_evdf_propagates_signal_background_and_cross_covariance():
    wavelength, signal, background = _case()
    out = _run(
        wavelength, signal, background,
        signal_uncertainty=np.full(signal.size, 0.5),
        background_uncertainty=np.full(signal.size, 0.2),
        signal_background_cross_covariance=np.eye(signal.size) * 0.05,
    )
    assert out["measurement_uncertainty_model"] == "joint_signal_background_covariance"
    assert out["background_uncertainty_included"] is True
    assert out["signal_background_cross_covariance_included"] is True
    assert "Cov(signal-background)" in out["residual_covariance_interpretation"]


def test_evdf_rejects_cross_covariance_without_both_marginal_models():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="requires both signal and background"):
        _run(
            wavelength, signal, background,
            signal_uncertainty=np.full(signal.size, 0.5),
            signal_background_cross_covariance=np.eye(signal.size) * 0.01,
        )


def test_evdf_rejects_non_positive_definite_combined_background_covariance():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="joint signal/background covariance"):
        _run(
            wavelength, signal, background,
            signal_uncertainty=np.full(signal.size, 0.1),
            background_uncertainty=np.full(signal.size, 1.0),
            signal_background_cross_covariance=np.eye(signal.size) * 0.6,
        )



def test_evdf_reports_regularization_sensitivity_across_gcv_candidates():
    wavelength, signal, background = _case()
    result = reconstruct_thomson_evdf(
        wavelength, signal,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        regularization_strength=None,
        velocity_limit_m_s=8.0e4,
        velocity_points=161,
    )

    sensitivity = result["regularization_sensitivity"]
    assert sensitivity["status"] == "assessed_over_gcv_candidate_grid"
    assert 1 <= sensitivity["candidate_count"] <= 13
    assert len(sensitivity["candidate_results"]) == sensitivity["candidate_count"]
    assert sensitivity["selected_regularization_strength"] == pytest.approx(
        result["regularization_strength"]
    )
    selected = min(
        sensitivity["candidate_results"],
        key=lambda item: abs(
            np.log(item["regularization_strength"])
            - np.log(result["regularization_strength"])
        ),
    )
    assert selected["total_variation_distance_from_selected"] == pytest.approx(0.0, abs=1e-12)
    assert sensitivity["ranges"]["velocity_parallel_standard_deviation_m_s"]["maximum"] >= sensitivity["ranges"]["velocity_parallel_standard_deviation_m_s"]["minimum"]
    assert "not a measurement uncertainty" in sensitivity["interpretation"]


def test_evdf_explicit_regularization_reports_single_candidate_not_sensitivity_range():
    wavelength, signal, background = _case()
    result = reconstruct_thomson_evdf(
        wavelength, signal,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        regularization_strength=1e-3,
        velocity_limit_m_s=8.0e4,
        velocity_points=161,
    )
    sensitivity = result["regularization_sensitivity"]
    assert sensitivity["status"] == "single_explicit_regularization_value"
    assert sensitivity["candidate_count"] == 1
    assert sensitivity["ranges"]["total_variation_distance_from_selected"] == {
        "minimum": pytest.approx(0.0, abs=1e-12),
        "maximum": pytest.approx(0.0, abs=1e-12),
    }


def test_evdf_candidate_shape_parameters_use_only_mathematical_domain_bounds(monkeypatch):
    import tokamak_analysis.thomson_evdf as module

    wavelength, signal, background = _case()
    original = module.least_squares
    observed_bounds = []

    def recording_least_squares(*args, **kwargs):
        observed_bounds.append(kwargs["bounds"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "least_squares", recording_least_squares)
    _run(wavelength, signal, background)

    assert len(observed_bounds) == 3
    # Candidate order: Maxwellian, finite-variance Student-t, super-Gaussian.
    kappa_lower, kappa_upper = observed_bounds[1]
    super_lower, super_upper = observed_bounds[2]
    assert kappa_lower[3] > 2.0
    assert np.isinf(kappa_upper[3])
    assert super_lower[3] > 0.0
    assert np.isinf(super_upper[3])


def test_evdf_uses_midpoint_finite_volume_bins_on_nonuniform_wavelength_grid():
    from tokamak_analysis.thomson_evdf import _wavelength_bin_widths

    wavelength = np.array([499.80, 499.86, 499.95, 500.00, 500.08, 500.20])
    expected = np.array([0.03, 0.075, 0.07, 0.065, 0.10, 0.06])
    np.testing.assert_allclose(_wavelength_bin_widths(wavelength), expected, rtol=0, atol=1e-14)

    # The endpoint bins are half the adjacent interval; all bins are positive
    # and the measure is valid for the non-uniform calibrated wavelength axis.
    widths = _wavelength_bin_widths(wavelength)
    assert np.all(widths > 0.0)
    assert widths[0] == pytest.approx(0.5 * (wavelength[1] - wavelength[0]))
    assert widths[-1] == pytest.approx(0.5 * (wavelength[-1] - wavelength[-2]))


def test_evdf_wavelength_bin_widths_reject_nonmonotonic_axis():
    from tokamak_analysis.thomson_evdf import _wavelength_bin_widths

    with pytest.raises(ValueError, match="strictly increasing"):
        _wavelength_bin_widths(np.array([500.0, 500.1, 500.05]))


def test_projected_evdf_is_exposed_through_public_physics_api_and_dashboard():
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    model = "thomson_projected_evdf"
    assert model in MODEL_LABELS
    configuration = EXAMPLES[model]
    assert "model_discrepancy_covariance" in configuration
    assert "systematic_covariance" in configuration
    result = run_advanced_thomson_physics(model, **configuration)
    assert result["public_physics_entrypoint"] == model
    assert result["automatic_model_substitution"] is False
    assert result["scattering_vector_projection_only"] is True
    assert result["absolute_density_identified"] is False
    assert result["evdf_parallel_s_per_m"].shape == result["velocity_parallel_m_s"].shape
    assert np.trapz(result["evdf_parallel_s_per_m"], result["velocity_parallel_m_s"]) == pytest.approx(1.0, rel=2e-5)


def test_relativistic_projected_evdf_and_psf_profile_are_dashboard_api_routes():
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    relativistic = "thomson_relativistic_projected_evdf"
    assert relativistic in MODEL_LABELS
    rel = run_advanced_thomson_physics(relativistic, **EXAMPLES[relativistic])
    assert rel["public_physics_entrypoint"] == relativistic
    assert rel["absolute_density_inferred"] is False
    assert "no Compton recoil" in " ".join(rel["assumptions"])

    profile = "thomson_psf_profile_inverse"
    assert profile in MODEL_LABELS
    spatial = run_advanced_thomson_physics(profile, **EXAMPLES[profile])
    assert spatial["public_physics_entrypoint"] == profile
    assert spatial["spatial_mixing_matrix"].shape == (5, 5)
    assert spatial["regularization_strength"] == pytest.approx(0.01)
    assert spatial["experimental_validation"] == "not_performed"


def test_evdf_adds_independent_model_discrepancy_and_systematic_covariance():
    wavelength, signal, background = _case()
    n = signal.size
    measurement = np.diag(np.full(n, 1.0**2))
    discrepancy = 0.2**2 * np.eye(n)
    systematic = 0.3**2 * np.eye(n)
    out = _run(
        wavelength, signal, background,
        signal_covariance=measurement,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
    )
    np.testing.assert_allclose(out["model_discrepancy_covariance"], discrepancy)
    np.testing.assert_allclose(out["systematic_covariance"], systematic)
    assert out["model_discrepancy_covariance_included"] is True
    assert out["systematic_covariance_included"] is True
    assert out["evdf_covariance_available"] is True
    assert "mutually independent" in out["additive_covariance_independence_assumption"]


def test_evdf_allows_positive_definite_discrepancy_only_weighting():
    wavelength, signal, background = _case()
    n = signal.size
    discrepancy = 0.5**2 * np.eye(n)
    out = _run(
        wavelength, signal, background,
        model_discrepancy_covariance=discrepancy,
    )
    assert out["measurement_uncertainty_model"] == "model_discrepancy_only_weighting"
    assert out["evdf_covariance_available"] is True


def test_evdf_rejects_indefinite_systematic_or_discrepancy_covariance():
    wavelength, signal, background = _case()
    invalid = np.eye(signal.size)
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _run(wavelength, signal, background, systematic_covariance=invalid)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _run(wavelength, signal, background, model_discrepancy_covariance=invalid)


def test_evdf_covariance_psd_validation_is_scale_invariant_for_small_si_units():
    wavelength, signal, background = _case()
    invalid = np.zeros((signal.size, signal.size))
    invalid[0, 0] = -1.0e-24
    invalid[1, 1] = 1.0e-24
    with pytest.raises(ValueError, match="positive semidefinite"):
        _run(wavelength, signal, background, systematic_covariance=invalid)


def test_evdf_allows_systematic_only_weighting_with_explicit_provenance():
    wavelength, signal, background = _case()
    systematic = 0.4**2 * np.eye(signal.size)
    out = _run(
        wavelength, signal, background,
        systematic_covariance=systematic,
    )
    assert out["measurement_uncertainty_model"] == "systematic_only_weighting"
    assert out["systematic_covariance_included"] is True
    assert out["model_discrepancy_covariance_included"] is False
