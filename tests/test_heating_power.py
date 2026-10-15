import numpy as np
from tis.heating_power import analyze_electron_ion_heating_power


def test_heating_power_partition():
    r = analyze_electron_ion_heating_power([100, 200], electron_fraction=0.7, ion_fraction=0.3)
    np.testing.assert_allclose(r['electron_heating_power_W'], [70, 140])
    np.testing.assert_allclose(r['ion_heating_power_W'], [30, 60])
    np.testing.assert_allclose(r['power_partition_residual_W'], 0)
