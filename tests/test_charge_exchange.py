import numpy as np
from tis.charge_exchange import analyze_charge_exchange_spectroscopy


def test_charge_exchange_spectroscopy_applies_optional_corrections():
    x = np.linspace(499, 501, 401)
    y = 1 + 20*np.exp(-0.5*((x-500)/0.04)**2)
    result = analyze_charge_exchange_spectroscopy(x, y, rest_wavelength_nm=500, ion_mass_amu=12,
        detector_response_factor=2, neutral_beam_overlap_factor=5)
    assert result['detector_response_corrected_signal'] > 0
    assert result['beam_overlap_normalized_signal'] == result['detector_response_corrected_signal']/5
