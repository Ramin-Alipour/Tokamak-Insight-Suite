import numpy as np

from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution


def _canonical_fit():
    return {
        "model": "non_collective_maxwellian",
        "electron_temperature_eV": 100.0,
        "electron_velocity_projection_m_s": 0.0,
    }


def test_distribution_gaussian_fit_is_returned_in_measured_intensity_units():
    wavelength = np.linspace(528.0, 536.0, 161)
    background = 15.0 + 0.4 * (wavelength - 532.0)
    signal = (
        750.0 * np.exp(-0.5 * ((wavelength - 532.15) / 0.55) ** 2)
        + background
    )
    result = analyze_thomson_spectral_distribution(
        wavelength,
        signal, physical_parameters=_canonical_fit(), 
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
    )
    fitted = np.asarray(result["fitted_spectrum"])
    residual = np.asarray(result["residual"])
    assert fitted.shape == signal.shape
    np.testing.assert_allclose(residual, signal - fitted, rtol=0.0, atol=1e-10)
    assert np.max(fitted) > 0.5 * np.max(signal)
    assert np.sqrt(np.mean(residual**2)) < 0.02 * np.max(signal)
