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
    with pytest.raises(ValueError, match="combined signal/background residual covariance"):
        _run(
            wavelength, signal, background,
            signal_uncertainty=np.full(signal.size, 0.1),
            background_uncertainty=np.full(signal.size, 1.0),
            signal_background_cross_covariance=np.eye(signal.size) * 0.2,
        )
