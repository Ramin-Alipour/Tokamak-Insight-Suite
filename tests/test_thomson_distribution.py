import numpy as np
import pytest

from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution


def _spectrum(sigma=0.18):
    x = np.linspace(499.0, 501.0, 501)
    y = 1000.0 * np.exp(-0.5 * ((x - 500.0) / sigma) ** 2) + 12.0 + 0.8 * (x - 500.0)
    return x, y


def test_spectral_distribution_recovers_gaussian_width_and_reports_shape():
    x, y = _spectrum()
    out = analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06,
                                                signal_uncertainty=np.full(x.size, 1.0))
    expected = np.sqrt(0.18**2 - 0.06**2)
    assert out["instrument_width_resolved"] is True
    assert out["instrument_corrected_sigma_nm"] == pytest.approx(expected, rel=0.03)
    assert out["spectral_centroid_nm"] == pytest.approx(500.0, abs=0.01)
    assert abs(out["spectral_skewness"]) < 0.03
    assert out["spectral_excess_kurtosis"] == pytest.approx(0.0, abs=0.1)
    assert out["reduced_chi_square"] is not None
    assert out["residual"].shape == y.shape


def test_spectral_distribution_flags_unresolved_instrument_width():
    x, y = _spectrum(sigma=0.03)
    out = analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06)
    assert out["instrument_width_resolved"] is False
    assert out["instrument_corrected_sigma_nm"] is None


@pytest.mark.parametrize("x,y,kwargs,match", [
    ([1, 2, 2, 4, 5, 6, 7, 8, 9], [1]*9, {"instrument_sigma_nm": 0.1}, "strictly increasing"),
    (np.arange(10), np.ones(9), {"instrument_sigma_nm": 0.1}, "equal lengths"),
    (np.arange(9), np.ones(9), {"instrument_sigma_nm": 0}, "positive and finite"),
    (np.arange(9), np.ones(9), {"instrument_sigma_nm": 0.1}, "positive spectral area"),
])
def test_spectral_distribution_rejects_invalid_inputs(x, y, kwargs, match):
    with pytest.raises(ValueError, match=match):
        analyze_thomson_spectral_distribution(x, y, **kwargs)


def test_spectral_distribution_rejects_invalid_uncertainty_and_background_fraction():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="strictly positive"):
        analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06,
                                              signal_uncertainty=np.zeros_like(y))
    with pytest.raises(ValueError, match="background_fraction"):
        analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06,
                                              background_fraction=0.9)

@pytest.mark.parametrize("bad_values", [
    [True] * 9,
    ["1"] * 9,
])
def test_spectral_distribution_rejects_boolean_and_string_arrays(bad_values):
    with pytest.raises(ValueError, match="real numeric values"):
        analyze_thomson_spectral_distribution(
            bad_values, np.ones(9), instrument_sigma_nm=0.1
        )


@pytest.mark.parametrize("kwargs", [
    {"instrument_sigma_nm": True},
    {"instrument_sigma_nm": "0.1"},
    {"instrument_sigma_nm": 0.1, "background_fraction": True},
    {"instrument_sigma_nm": 0.1, "background_fraction": "0.1"},
])
def test_spectral_distribution_rejects_boolean_and_string_scalar_parameters(kwargs):
    x, y = _spectrum()
    with pytest.raises(ValueError):
        analyze_thomson_spectral_distribution(x, y, **kwargs)


def test_moments_are_correct_on_nonuniform_wavelength_grid():
    # A nonuniform grid must not bias moments toward densely sampled regions.
    left = np.linspace(499.0, 499.8, 90, endpoint=False)
    center = np.linspace(499.8, 500.2, 401, endpoint=False)
    right = np.linspace(500.2, 501.0, 90)
    x = np.unique(np.concatenate([left, center, right]))
    y = 1000.0 * np.exp(-0.5 * ((x - 500.0) / 0.18) ** 2) + 12.0
    out = analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06)
    assert out["spectral_centroid_nm"] == pytest.approx(500.0, abs=0.003)
    assert out["instrument_corrected_sigma_nm"] == pytest.approx(
        np.sqrt(0.18**2 - 0.06**2), rel=0.04
    )
    assert out["wavelength_quadrature_weights_nm"].shape == x.shape
    assert out["integrated_spectral_measure"].shape == x.shape


def test_spectral_distribution_handles_constant_signal_without_nonfinite_json_value():
    x = np.linspace(499.0, 501.0, 101)
    y = np.full_like(x, 7.0)
    # A zero net spectral area must be rejected clearly rather than producing NaN metrics.
    with pytest.raises(ValueError, match="positive spectral area"):
        analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.05)


def test_spectral_distribution_validates_uncertainty_length_and_nonfinite_values():
    x, y = _spectrum()
    with pytest.raises(ValueError, match="same length"):
        analyze_thomson_spectral_distribution(
            x, y, instrument_sigma_nm=0.06, signal_uncertainty=np.ones(10)
        )
    bad = y.copy()
    bad[20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_thomson_spectral_distribution(x, bad, instrument_sigma_nm=0.06)


def test_spectral_distribution_result_is_json_safe_for_degenerate_r_squared():
    # A valid peaked signal has nonzero variance and finite fit metrics.
    x, y = _spectrum()
    out = analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06)
    assert np.isfinite(out["r_squared"])

def test_complete_result_can_be_exported_as_strict_json_including_spectral_arrays():
    import json

    x, y = _spectrum()
    out = analyze_thomson_spectral_distribution(
        x, y, instrument_sigma_nm=0.06,
        signal_uncertainty=np.full(x.size, 1.0),
    )
    payload = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in out.items()
    }
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    for key in (
        "wavelength_nm", "signal", "background", "background_subtracted_signal",
        "positive_spectral_weights", "wavelength_quadrature_weights_nm",
        "integrated_spectral_measure", "fitted_spectrum", "residual",
    ):
        assert len(decoded[key]) == x.size
    assert decoded["gaussian_fit_parameters"]["center_nm"] == pytest.approx(500.0, abs=0.01)



def test_fit_uncertainties_and_goodness_metrics_remain_strict_json_safe():
    import json

    x, y = _spectrum()
    out = analyze_thomson_spectral_distribution(
        x, y, instrument_sigma_nm=0.06,
        signal_uncertainty=np.full(x.size, 1.0),
    )
    # Covariance-derived values may be unavailable for ill-conditioned fits, but
    # the public result must use JSON null rather than NaN/Infinity.
    encoded = json.dumps(out, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value, allow_nan=False)
    assert json.loads(encoded)["chi_square"] is not None


def test_thomson_distribution_is_not_a_separate_canonical_analysis_entry():
    from tokamak_analysis.analysis_registry import ANALYSES, get_analysis

    assert len(ANALYSES) == 80
    with pytest.raises(KeyError, match="Unknown analysis ID: 81"):
        get_analysis(81)
    # The numerical function remains tested directly below; it is not counted
    # as a separate diagnostic/analysis entry in the canonical 80-item registry.


@pytest.mark.parametrize("scale", [1e-20, 1e-12, 1.0, 1e12])
def test_spectral_distribution_is_invariant_to_signal_units(scale):
    x, y = _spectrum()
    baseline = analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06)
    scaled = analyze_thomson_spectral_distribution(x, y * scale, instrument_sigma_nm=0.06)
    assert scaled["spectral_centroid_nm"] == pytest.approx(baseline["spectral_centroid_nm"], abs=1e-8)
    assert scaled["gaussian_fit_parameters"]["center_nm"] == pytest.approx(
        baseline["gaussian_fit_parameters"]["center_nm"], abs=1e-6
    )
    assert scaled["gaussian_fit_parameters"]["amplitude"] == pytest.approx(
        baseline["gaussian_fit_parameters"]["amplitude"] * scale, rel=1e-6
    )
    assert scaled["observed_sigma_nm"] == pytest.approx(baseline["observed_sigma_nm"], rel=1e-8)
    assert scaled["instrument_corrected_sigma_nm"] == pytest.approx(
        baseline["instrument_corrected_sigma_nm"], rel=1e-7
    )
    assert scaled["integrated_positive_spectral_signal"] == pytest.approx(
        baseline["integrated_positive_spectral_signal"] * scale, rel=1e-7
    )


def test_spectral_distribution_reports_optimizer_failure_as_actionable_value_error(monkeypatch):
    import tokamak_analysis.thomson_distribution as module

    x, y = _spectrum()

    def fail_fit(*args, **kwargs):
        raise RuntimeError("synthetic optimizer non-convergence")

    monkeypatch.setattr(module, "curve_fit", fail_fit)
    with pytest.raises(ValueError, match="Gaussian spectral-shape fit failed"):
        module.analyze_thomson_spectral_distribution(x, y, instrument_sigma_nm=0.06)
