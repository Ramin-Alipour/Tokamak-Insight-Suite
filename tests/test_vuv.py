import numpy as np
from tis.vuv import analyze_vuv_spectrum


def test_vuv_response_corrected_line_signals():
    x = np.linspace(100, 104, 801)
    y = 1 + 20*np.exp(-0.5*((x-101)/0.03)**2) + 10*np.exp(-0.5*((x-103)/0.04)**2)
    result = analyze_vuv_spectrum(x, y, [
        {'label': 'A', 'minimum_nm': 100.7, 'maximum_nm': 101.3},
        {'label': 'B', 'minimum_nm': 102.6, 'maximum_nm': 103.4},
    ], detector_response={'A': 2.0, 'B': 1.0})
    assert result['response_corrected_signals'][0]['corrected_signal'] > 0
    assert len(result['corrected_line_ratios']) == 1
