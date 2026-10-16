import numpy as np
from tis.fida import analyze_fida_spectrum


def test_fida_feature_fit_and_velocity_proxy():
    x = np.linspace(655, 657, 401)
    y = 1 + 10*np.exp(-0.5*((x-656.4)/0.05)**2)
    r = analyze_fida_spectrum(x, y, fast_ion_region_nm=(656.2, 656.6), reference_wavelength_nm=656.28, detector_response_factor=2)
    assert r['integrated_feature_signal'] > 0
    assert r['detector_response_corrected_signal'] > 0
