import numpy as np
import pytest

from tokamak_analysis.thomson_inference import _fit_noncollective, _fit_kappa


@pytest.mark.parametrize("model", ["maxwellian", "kappa"])
def test_fitted_spectrum_and_residual_are_in_measured_intensity_units(model):
    x = np.linspace(496.0, 568.0, 481)
    sigma = 2.0 * 532.0 * np.sqrt(100.0 / 510998.95069) * np.sin(np.pi / 4)

    if model == "maxwellian":
        signal = (
            1200.0 * np.exp(-0.5 * ((x - 532.0) / sigma) ** 2)
            + 25.0 + 0.1 * (x - 532.0)
        )
        fit = _fit_noncollective(
            x, signal, 532.0, np.pi / 2, np.eye(x.size), None
        )
    else:
        kappa = 4.0
        df = 2.0 * kappa - 1.0
        scale = sigma * np.sqrt((df - 2.0) / df)
        signal = (
            900.0 * (1.0 + ((x - 532.0) / scale) ** 2 / df) ** (-kappa)
            + 20.0
        )
        fit = _fit_kappa(x, signal, 532.0, np.pi / 2, np.eye(x.size), None)

    fitted = np.asarray(fit["fitted_spectrum"])
    residual = np.asarray(fit["residual"])
    assert fitted.shape == signal.shape
    assert residual.shape == signal.shape
    np.testing.assert_allclose(residual, signal - fitted, rtol=0.0, atol=1e-10)
    assert np.sqrt(np.mean(residual**2)) < 0.02 * np.max(signal)
    assert np.max(fitted) > 0.5 * np.max(signal)
