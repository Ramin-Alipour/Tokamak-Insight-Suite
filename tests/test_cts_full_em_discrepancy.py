import numpy as np
import pytest

from tokamak_analysis.cts_full_em import _prepare_full_em_cts_spectral_whitener


def test_full_em_cts_adds_independent_discrepancy_to_measurement_covariance():
    measurement = np.diag(np.full(5, 0.04))
    discrepancy = np.full((5, 5), 0.002)
    np.fill_diagonal(discrepancy, 0.01)
    whiten, effective, status = _prepare_full_em_cts_spectral_whitener(
        5,
        spectral_covariance=measurement,
        model_discrepancy_covariance=discrepancy,
    )
    np.testing.assert_allclose(effective, measurement + discrepancy)
    assert status == "measurement_covariance_plus_independent_correlated_model_discrepancy"
    residual = np.arange(1.0, 6.0)
    assert np.all(np.isfinite(whiten(residual)))


def test_full_em_cts_accepts_discrepancy_only_without_inventing_measurement_noise():
    discrepancy = np.diag(np.linspace(0.1, 0.5, 6))
    whiten, effective, status = _prepare_full_em_cts_spectral_whitener(
        6, model_discrepancy_covariance=discrepancy
    )
    np.testing.assert_array_equal(effective, discrepancy)
    assert status == "model_discrepancy_covariance_only_no_measurement_uncertainty_supplied"
    assert np.all(np.isfinite(whiten(np.ones(6))))


def test_full_em_cts_rejects_non_psd_or_mismatched_discrepancy():
    with pytest.raises(ValueError, match="positive semidefinite"):
        _prepare_full_em_cts_spectral_whitener(
            4, model_discrepancy_covariance=np.diag([1.0, 1.0, 1.0, -0.1])
        )
    with pytest.raises(ValueError, match="match the measured frequency grid"):
        _prepare_full_em_cts_spectral_whitener(
            4, model_discrepancy_covariance=np.eye(3)
        )


def test_full_em_cts_does_not_silently_regularize_singular_total_covariance():
    singular = np.ones((4, 4))
    with pytest.raises(ValueError, match="not silently regularized"):
        _prepare_full_em_cts_spectral_whitener(
            4, model_discrepancy_covariance=singular
        )


def test_full_em_cts_public_inverse_exposes_discrepancy_argument():
    from inspect import signature
    from tokamak_analysis.cts_full_em import analyze_full_em_cts_spectrum

    assert "model_discrepancy_covariance" in signature(
        analyze_full_em_cts_spectrum
    ).parameters


def test_full_em_cts_pointwise_measurement_error_combines_as_diagonal_covariance():
    sigma = np.linspace(0.1, 0.3, 5)
    discrepancy = np.eye(5) * 0.02
    _, effective, status = _prepare_full_em_cts_spectral_whitener(
        5,
        signal_uncertainty=sigma,
        model_discrepancy_covariance=discrepancy,
    )
    np.testing.assert_allclose(effective, np.diag(sigma**2) + discrepancy)
    assert status == "measurement_covariance_plus_independent_correlated_model_discrepancy"


def test_cts_registry_declares_discrepancy_covariance_for_both_models():
    import json
    from pathlib import Path

    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item for item in registry
        if item.get("diagnostic") == "Collective Thomson / coherent scattering (CTS)"
    )
    spec = next(
        item for item in entry["execution_contract"]["input_specs"]
        if item["name"] == "model_discrepancy_covariance"
    )
    assert "Electrostatic and Full-EM" in spec["description"]
