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
    velocity_grid = np.asarray(configuration["velocity_grid_m_s"], dtype=float)
    assert np.all(np.diff(velocity_grid) > 0.0)
    assert not np.allclose(np.diff(velocity_grid), np.diff(velocity_grid)[0])
    assert "model_discrepancy_covariance" in configuration
    assert "systematic_covariance" in configuration
    result = run_advanced_thomson_physics(model, **configuration)
    assert result["public_physics_entrypoint"] == model
    assert result["automatic_model_substitution"] is False
    assert result["scattering_vector_projection_only"] is True
    assert result["absolute_density_identified"] is False
    assert result["evdf_parallel_s_per_m"].shape == result["velocity_parallel_m_s"].shape
    assert np.trapz(result["evdf_parallel_s_per_m"], result["velocity_parallel_m_s"]) == pytest.approx(1.0, rel=2e-5)
    assert result["velocity_moment_covariance"].shape == (6, 6)
    assert result["velocity_moment_standard_uncertainty"] is not None
    assert result["velocity_moment_uncertainty_status"] == (
        "delta_method_propagated_from_normalized_evdf_covariance"
    )


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

def test_evdf_physical_velocity_moment_covariance_is_propagated_and_correlated():
    wavelength, signal, background = _case()
    out = _run(
        wavelength, signal, background,
        signal_uncertainty=np.full(signal.size, 0.5),
    )
    covariance = out["velocity_moment_covariance"]
    order = out["velocity_moment_covariance_order"]
    assert covariance.shape == (6, 6)
    assert order == [
        "mean_velocity_parallel_m_s",
        "velocity_parallel_variance_m2_s2",
        "velocity_parallel_standard_deviation_m_s",
        "directional_equivalent_temperature_eV",
        "skewness",
        "excess_kurtosis",
    ]
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-10, atol=1e-18)
    assert np.linalg.eigvalsh(covariance)[0] >= -1e-8 * max(
        np.max(np.abs(covariance)), 1.0
    )
    standard = out["velocity_moment_standard_uncertainty"]
    assert standard is not None
    assert all(np.isfinite(value) and value >= 0.0 for value in standard.values())
    electron_mass = 9.1093837139e-31
    boltzmann = 1.380649e-23
    elementary_charge = 1.602176634e-19
    expected_temperature_variance = (
        electron_mass / (boltzmann * elementary_charge)
    ) ** 2 * covariance[1, 1]
    assert covariance[3, 3] == pytest.approx(
        expected_temperature_variance, rel=2e-8, abs=1e-18
    )
    assert out["velocity_moment_uncertainty_status"] == (
        "delta_method_propagated_from_normalized_evdf_covariance"
    )



def test_evdf_normalization_jacobian_supports_nonuniform_velocity_grid():
    from tokamak_analysis.thomson_evdf import _normalize_evdf_coefficients

    velocity = np.array([-4.0, -2.5, -1.0, -0.2, 0.4, 1.8, 4.5])
    coefficients = np.array([0.3, 0.8, 1.4, 2.0, 1.2, 0.7, 0.2])
    distribution, analytic = _normalize_evdf_coefficients(coefficients, velocity)
    assert np.trapz(distribution, velocity) == pytest.approx(1.0, abs=1e-13)
    numerical = np.empty_like(analytic)
    step = 1e-6
    for index in range(coefficients.size):
        plus, minus = coefficients.copy(), coefficients.copy()
        plus[index] += step
        minus[index] -= step
        numerical[:, index] = (
            _normalize_evdf_coefficients(plus, velocity)[0]
            - _normalize_evdf_coefficients(minus, velocity)[0]
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-7, atol=2e-9)


def test_projected_evdf_reconstruction_accepts_nonuniform_velocity_grid():
    wavelength, signal, background = _case()
    coordinate = np.linspace(-1.0, 1.0, 161)
    velocity = 8.0e4 * coordinate**3
    out = _run(
        wavelength, signal, background,
        velocity_grid_m_s=velocity,
    )
    reconstructed = out["evdf_parallel_s_per_m"]
    assert np.all(np.isfinite(reconstructed))
    assert np.all(reconstructed >= 0.0)
    assert np.trapz(reconstructed, velocity) == pytest.approx(1.0, rel=2e-5)
    assert out["velocity_parallel_m_s"].shape == velocity.shape


def test_nonuniform_velocity_curvature_operator_is_exact_for_quadratics():
    from tokamak_analysis.thomson_evdf import _nonuniform_second_derivative_matrix

    velocity = np.array([-5.0, -3.2, -1.1, -0.4, 0.3, 2.0, 5.7])
    operator = _nonuniform_second_derivative_matrix(velocity)
    np.testing.assert_allclose(operator @ np.ones_like(velocity), 0.0, atol=1e-13)
    np.testing.assert_allclose(operator @ velocity, 0.0, atol=1e-13)
    np.testing.assert_allclose(operator @ (velocity**2), 2.0, rtol=1e-12, atol=1e-12)



def test_evdf_reports_akaike_model_selection_and_between_model_uncertainty():
    wavelength, signal, background = _case()
    result = _run(
        wavelength, signal, background,
        signal_uncertainty=np.full(signal.size, 1.0),
    )
    selection = result["parametric_model_selection_uncertainty"]
    assert selection["status"] == "akaike_weighted_model_uncertainty"
    assert set(selection["weights"]) == {"maxwellian", "kappa_student_t", "super_gaussian"}
    assert sum(selection["weights"].values()) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= weight <= 1.0 for weight in selection["weights"].values())
    assert all(value >= 0.0 for value in selection["delta_aic"].values())
    mean = selection["model_averaged"]["mean_velocity_parallel_m_s"]
    scale = selection["model_averaged"]["scale_velocity_m_s"]
    assert mean["candidate_count"] == 3
    assert scale["candidate_count"] == 3
    assert mean["between_model_standard_uncertainty"] >= 0.0
    assert scale["between_model_standard_uncertainty"] >= 0.0
    assert mean["total_model_averaged_standard_uncertainty"] is not None
    assert "not Bayesian posterior" in selection["interpretation"]


def test_evdf_model_selection_excludes_nonconverged_candidates(monkeypatch):
    from types import SimpleNamespace
    import tokamak_analysis.thomson_evdf as evdf

    wavelength, signal, background = _case()
    original = evdf.least_squares
    calls = {"count": 0}

    def one_failed_candidate(objective, initial, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            return SimpleNamespace(
                success=False, x=np.asarray(initial, dtype=float),
                message="forced candidate failure",
            )
        return original(objective, initial, **kwargs)

    monkeypatch.setattr(evdf, "least_squares", one_failed_candidate)
    result = _run(wavelength, signal, background)
    selection = result["parametric_model_selection_uncertainty"]
    assert selection["status"] == "akaike_weighted_model_uncertainty"
    assert "kappa_student_t" not in selection["weights"]
    assert sum(selection["weights"].values()) == pytest.approx(1.0, abs=1e-12)



def test_evdf_forward_operator_integrates_velocity_density_with_nonuniform_quadrature():
    from tokamak_analysis.thomson_evdf import (
        _projected_evdf_forward_matrix,
        _velocity_quadrature_weights,
    )

    wavelength = np.linspace(499.0, 501.0, 801)
    velocity = np.array([-2.0e5, -1.3e5, -7.0e4, -2.0e4, 1.0e4, 5.0e4, 1.2e5, 2.0e5])
    response = np.eye(wavelength.size)
    forward, metadata = _projected_evdf_forward_matrix(
        wavelength, velocity, response, 500.0, 2.0 * np.sin(np.pi / 4.0)
    )
    wavelength_widths = metadata["wavelength_bin_widths_nm"]
    quadrature = _velocity_quadrature_weights(velocity)

    # A unit-integral line kernel has spectral area one; after discretizing
    # int f(v) K(lambda|v) dv, each density column must carry its dv weight.
    integrated_columns = np.sum(forward * wavelength_widths[:, None], axis=0)
    np.testing.assert_allclose(integrated_columns, quadrature, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(
        metadata["column_integrated_spectral_power_per_unit_evdf"],
        quadrature, rtol=2e-12, atol=1e-10,
    )


def test_evdf_forward_operator_preserves_velocity_integrated_power_for_nonuniform_density():
    from tokamak_analysis.thomson_evdf import _projected_evdf_forward_matrix

    wavelength = np.linspace(499.0, 501.0, 1001)
    velocity = np.array([-2.0e5, -1.4e5, -8.0e4, -3.0e4, 2.0e4, 9.0e4, 1.5e5, 2.0e5])
    density = np.exp(-0.5 * (velocity / 8.0e4) ** 2)
    forward, metadata = _projected_evdf_forward_matrix(
        wavelength, velocity, np.eye(wavelength.size), 500.0,
        2.0 * np.sin(np.pi / 4.0),
    )
    predicted = forward @ density
    spectral_power = np.sum(predicted * metadata["wavelength_bin_widths_nm"])
    velocity_power = np.dot(density, metadata["velocity_quadrature_weights_m_s"])
    assert spectral_power == pytest.approx(velocity_power, rel=2e-12)



def test_evdf_forward_operator_conserves_quadrature_with_measured_response_and_nonuniform_wavelength_bins():
    from tokamak_analysis.thomson_evdf import _projected_evdf_forward_matrix

    wavelength = np.array([
        499.0, 499.08, 499.17, 499.29, 499.42, 499.56, 499.71,
        499.87, 500.0, 500.13, 500.29, 500.44, 500.58, 500.71,
        500.83, 500.92, 501.0,
    ])
    velocity = np.array([-2.0e5, -1.2e5, -6.0e4, -1.0e4, 4.0e4, 1.1e5, 2.0e5])
    n = wavelength.size
    response = np.zeros((n, n))
    for column in range(n):
        response[column, column] += 0.7
        if column > 0:
            response[column - 1, column] += 0.2
        else:
            response[column, column] += 0.2
        if column + 1 < n:
            response[column + 1, column] += 0.1
        else:
            response[column, column] += 0.1

    forward, metadata = _projected_evdf_forward_matrix(
        wavelength, velocity, response, 500.0, 2.0 * np.sin(np.pi / 4.0)
    )
    integrated = np.sum(
        forward * metadata["wavelength_bin_widths_nm"][:, None], axis=0
    )
    np.testing.assert_allclose(
        integrated, metadata["velocity_quadrature_weights_m_s"],
        rtol=2e-12, atol=1e-10,
    )


def test_advanced_dashboard_evdf_route_exposes_velocity_integral_discretization():
    from tis.ui.advanced_thomson_physics_page import EXAMPLES
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "thomson_projected_evdf", **EXAMPLES["thomson_projected_evdf"]
    )
    assert result["velocity_integral_discretization"] == (
        "trapezoidal_finite_volume_measure_on_nodal_EVDF_density"
    )
    assert result["forward_operator_column_integrated_spectral_power_per_unit_evdf"].shape == (
        result["velocity_parallel_m_s"].shape
    )


def test_regularized_evdf_covariance_uses_frequentist_sandwich_not_posterior_inverse():
    from tokamak_analysis.thomson_evdf import (
        _regularized_active_set_measurement_covariance,
    )

    design = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    penalty = np.array([[1.0, -1.0]])
    strength = 3.0
    information = design.T @ design
    hessian = information + strength * (penalty.T @ penalty)
    inverse = np.linalg.inv(hessian)
    expected = inverse @ information @ inverse
    posterior_covariance = inverse

    actual = _regularized_active_set_measurement_covariance(
        design, penalty, strength
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert not np.allclose(actual, posterior_covariance)
    np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1e-14)
    assert np.linalg.eigvalsh(actual)[0] >= -1e-12


def test_regularized_evdf_covariance_scales_with_whitened_residual_variance():
    from tokamak_analysis.thomson_evdf import (
        _regularized_active_set_measurement_covariance,
    )

    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    penalty = np.array([[1.0, -1.0]])
    base = _regularized_active_set_measurement_covariance(
        design, penalty, 0.7, residual_variance=1.0
    )
    scaled = _regularized_active_set_measurement_covariance(
        design, penalty, 0.7, residual_variance=4.0
    )
    np.testing.assert_allclose(scaled, 4.0 * base, rtol=1e-12, atol=1e-14)
