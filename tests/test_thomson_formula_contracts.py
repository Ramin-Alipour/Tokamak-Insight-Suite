import numpy as np
import math
import pytest
from tokamak_analysis.formula_library import thomson_electron_temperature_from_spectral_width, thomson_velocity_projection, thomson_density_from_scattered_power

def test_thomson_temperature_formula_reference():
    out=thomson_electron_temperature_from_spectral_width(0.2,532.0,math.pi/2,sigma_intrinsic_sigma_nm=0.002,sigma_laser_wavelength_nm=0.001,sigma_scattering_angle_rad=1e-4)
    expected=(0.2/532.0)**2*510998.95069/(4*math.sin(math.pi/4)**2)
    assert out["electron_temperature_eV"]==pytest.approx(expected,rel=1e-12)
    assert out["uncertainty_electron_temperature_eV"]>0

def test_thomson_velocity_formula_reference():
    out=thomson_velocity_projection(532.01,532.0,math.pi/2)
    assert out["electron_velocity_projection_m_s"]==pytest.approx(-299792458.0*0.01/(2*532.0*math.sin(math.pi/4)),rel=1e-12)

def test_thomson_density_formula_reference():
    out=thomson_density_from_scattered_power(1e-9,1.0,0.1,math.pi/2,1e-3,optical_efficiency=0.5)
    assert out["electron_density_m3"]>0

def test_thomson_formulas_reject_invalid_physics():
    with pytest.raises(ValueError): thomson_electron_temperature_from_spectral_width(0.2,532.0,0.0)
    with pytest.raises(ValueError,match="superluminal"): thomson_velocity_projection(1500.0,500.0,math.pi/2)


def test_thomson_density_correlated_input_covariance():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power
    covariance = np.eye(6) * 1e-4
    out = density_from_thomson_scattered_power(
        1e-9, 1.0, 0.1, math.pi/2, 1e-3,
        optical_efficiency=0.5,
        relative_input_covariance=covariance,
    )
    assert out["electron_density_uncertainty_m3"] == pytest.approx(
        out["electron_density_m3"] * math.sqrt(6e-4), rel=1e-12
    )
    assert out["density_interpretation"] == "interaction_length_averaged_effective_density"
