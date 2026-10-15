import numpy as np
import pytest
from tis.ion_saturation import analyze_ion_saturation_current


def test_ion_saturation_current_flux_and_density_are_positive():
    result = analyze_ion_saturation_current([-0.01, 0.02], collection_area_m2=1e-5, ion_mass_amu=2, electron_temperature_eV=10)
    assert np.all(result['particle_flux_m-2_s-1'] > 0)
    assert np.all(result['ion_density_estimate_m-3'] > 0)
    assert result['ion_sound_speed_m_s'] > 0


def test_ion_saturation_requires_positive_area():
    with pytest.raises(ValueError, match='collection_area_m2'):
        analyze_ion_saturation_current([0.01], collection_area_m2=0, ion_mass_amu=2, electron_temperature_eV=10)
