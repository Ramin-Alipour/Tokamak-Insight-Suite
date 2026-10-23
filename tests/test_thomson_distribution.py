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
    (np.arange(9), np.ones(8), {"instrument_sigma_nm": 0.1}, "equal lengths"),
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
