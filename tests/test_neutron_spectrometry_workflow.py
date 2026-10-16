import numpy as np
from tis.neutron_spectrometry import analyze_neutron_spectrometry


def test_neutron_spectrum_unfolding():
    response = np.eye(3)
    result = analyze_neutron_spectrometry([1., 2., 3.], response, [1., 2., 3.], live_time_s=2, regularization=0)
    assert result['mean_neutron_energy_keV'] > 0
    assert result['unfolded_spectrum_counts_proxy'].shape == (3,)
