"""Independent analytic tests for the mapped formula-library contracts."""
import math
import numpy as np
import pytest
from tis.constants import E_CHARGE, EPS0, M_E
from tokamak_analysis.formula_library import (
    electron_plasma_frequency, debye_length, electron_cyclotron_frequency,
    ion_sound_speed, electron_thermal_pressure, ion_thermal_pressure,
    total_thermal_pressure, thermal_stored_energy, line_integrated_density,
)

def test_electron_plasma_frequency_independent_reference():
    n=1.0e19; expected=math.sqrt(n*E_CHARGE**2/(EPS0*M_E))
    r=electron_plasma_frequency(n,sigma_n_e_m3=1.0e17)
    assert r["omega_pe_rad_s"] == pytest.approx(expected,rel=1e-12)
    assert r["frequency_pe_Hz"] == pytest.approx(expected/(2*math.pi),rel=1e-12)
    assert r["uncertainty_omega_pe_rad_s"] == pytest.approx(expected*0.005,rel=1e-12)

def test_debye_length_independent_reference():
    T,n=100.0,1e19
    expected=math.sqrt(EPS0*E_CHARGE*T/(n*E_CHARGE**2))
    r=debye_length(T,n,sigma_temperature_eV=2.0,sigma_density_m3=1e17)
    assert r["debye_length_m"] == pytest.approx(expected,rel=1e-12)
    assert r["uncertainty_debye_length_m"] == pytest.approx(0.5*expected*math.sqrt((2/T)**2+(1e17/n)**2),rel=1e-12)

def test_electron_cyclotron_frequency_independent_reference():
    B=2.0; expected=E_CHARGE*B/M_E
    r=electron_cyclotron_frequency(B,sigma_magnetic_field_T=0.01)
    assert r["omega_ce_rad_s"] == pytest.approx(expected,rel=1e-12)
    assert r["frequency_ce_Hz"] == pytest.approx(expected/(2*math.pi),rel=1e-12)

def test_ion_sound_speed_independent_reference():
    Te,Ti,mi,gamma=20.0,10.0,2*1.66053906660e-27,1.0
    assert ion_sound_speed(Te,Ti,mi,gamma)["ion_sound_speed_m_s"] == pytest.approx(math.sqrt(E_CHARGE*(Te+gamma*Ti)/mi),rel=1e-12)

def test_electron_pressure_independent_reference():
    n,T=2e19,100.0; expected=n*E_CHARGE*T
    r=electron_thermal_pressure(n,T,sigma_density_m3=1e17,sigma_temperature_eV=1.0)
    assert r["electron_pressure_Pa"] == pytest.approx(expected,rel=1e-12)
    assert r["uncertainty_electron_pressure_Pa"] == pytest.approx(E_CHARGE*math.sqrt((T*1e17)**2+(n*1.0)**2),rel=1e-12)

def test_ion_pressure_multispecies_reference():
    r=ion_thermal_pressure([1e19,2e19],[10,20],sigma_species_density_m3=[1e17,2e17],sigma_species_temperature_eV=[1,2])
    assert r["ion_pressure_Pa"] == pytest.approx(E_CHARGE*(1e19*10+2e19*20),rel=1e-12)
    assert r["uncertainty_ion_pressure_Pa"] > 0

def test_total_pressure_reference():
    r=total_thermal_pressure(10.0,20.0,sigma_electron_pressure_Pa=1.0,sigma_ion_pressure_Pa=2.0)
    assert r["total_thermal_pressure_Pa"] == 30.0
    assert r["uncertainty_total_thermal_pressure_Pa"] == pytest.approx(math.sqrt(5))

def test_thermal_stored_energy_integral_reference():
    r=thermal_stored_energy([10,10,10],[0,1,2],sigma_pressure_Pa=[1,1,1])
    assert r["thermal_stored_energy_J"] == pytest.approx(30.0)
    assert r["uncertainty_thermal_stored_energy_J"] == pytest.approx(1.5*math.sqrt(0.25+1+0.25))

def test_line_integrated_density_reference():
    r=line_integrated_density([1e19,1e19,1e19],[0,0.5,1.0],sigma_density_m3=[1e17]*3)
    assert r["line_integrated_density_m-2"] == pytest.approx(1e19)
    assert r["uncertainty_line_integrated_density_m-2"] == pytest.approx(1e17*math.sqrt(0.375))

@pytest.mark.parametrize("call", [
    lambda: electron_plasma_frequency(0), lambda: debye_length(-1,1e19),
    lambda: electron_cyclotron_frequency(float("nan")), lambda: ion_sound_speed(1,-1,1),
    lambda: electron_thermal_pressure(1,-1), lambda: ion_thermal_pressure([1],[1,2]),
    lambda: total_thermal_pressure(-1,1), lambda: thermal_stored_energy([1,2],[1,0]),
    lambda: line_integrated_density([1,2],[0,0]),
])
def test_invalid_formula_inputs_raise_value_error(call):
    with pytest.raises(ValueError): call()
