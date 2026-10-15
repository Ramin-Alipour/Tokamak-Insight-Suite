import numpy as np
from tis.instrument_response import characterize_instrument_line_shape


def test_instrument_response_recovers_gaussian_width():
    x = np.linspace(499, 501, 501)
    y = 1 + 10*np.exp(-0.5*((x-500)/0.03)**2)
    result = characterize_instrument_line_shape(x, y, known_wavelength_nm=500)
    assert abs(result['instrumental_fwhm_nm'] - 2.354820045*0.03) < 1e-4
    assert result['resolving_power_lambda_over_delta_lambda'] > 1000
