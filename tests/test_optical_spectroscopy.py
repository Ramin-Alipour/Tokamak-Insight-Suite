import numpy as np
import pytest
from tis.optical_spectroscopy import analyze_optical_emission_spectrum


def test_optical_emission_fits_multiple_lines_and_velocity():
    x = np.linspace(499, 503, 801)
    y = 2 + 20*np.exp(-0.5*((x-500.5)/0.03)**2) + 15*np.exp(-0.5*((x-502.0)/0.05)**2)
    result = analyze_optical_emission_spectrum(x, y, [
        {'label': 'line A', 'minimum_nm': 500.2, 'maximum_nm': 500.8, 'rest_wavelength_nm': 500.5},
        {'label': 'line B', 'minimum_nm': 501.6, 'maximum_nm': 502.4},
    ])
    assert result['n_lines_fitted'] == 2
    assert abs(result['lines'][0]['line_of_sight_velocity_m_s']) < 100
    assert result['lines'][1]['fit']['FWHM'] > 0


def test_optical_emission_rejects_nonmonotonic_wavelength_axis():
    with pytest.raises(ValueError, match='strictly increasing'):
        analyze_optical_emission_spectrum([1, 2, 2, 4, 5, 6, 7], [1]*7, [{'minimum_nm': 1, 'maximum_nm': 5}])
