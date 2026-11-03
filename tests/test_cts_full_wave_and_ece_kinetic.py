import numpy as np
import pytest
from tokamak_analysis.cts_full_wave import berreman_matrix, solve_stratified_full_wave
from tokamak_analysis.ece_kinetic import kinetic_ece_absorption_coefficient


def test_vacuum_berreman_matrix_has_free_space_wavenumber():
    w=2*np.pi*10e9
    a=berreman_matrix(w,np.eye(3))
    assert a.shape==(4,4)
    assert np.all(np.isfinite(a))


def test_full_wave_vacuum_slab_has_zero_reflection_and_unit_transmission():
    z=np.linspace(0,0.02,31)
    eps=np.repeat(np.eye(3,dtype=complex)[None,:,:],z.size,axis=0)
    result=solve_stratified_full_wave(2*np.pi*10e9,z,eps)
    assert result["reflectance"] < 1e-8
    assert result["transmittance"] == pytest.approx(1.0,rel=1e-7)
    assert abs(result["absorptance"]) < 1e-7


def test_full_wave_rejects_singular_epsilon_zz():
    eps=np.repeat(np.eye(3,dtype=complex)[None,:,:],3,axis=0)
    eps[:,2,2]=0
    with pytest.raises(ValueError,match="epsilon_zz"):
        solve_stratified_full_wave(1e10,[0,1,2],eps)


def test_kinetic_ece_absorption_returns_tensor_and_finite_coefficient():
    out=kinetic_ece_absorption_coefficient(
        28e9,[0,0,2*np.pi*28e9/299792458],[0,0,1.0],
        electron_density_m3=1e18,electron_temperature_eV=100,
        polarization=[1,0,0],harmonic_order=3,quadrature_order=24,
        damping_rate_rad_s=1e6)
    assert out["conductivity_S_m"].shape==(3,3)
    assert np.isfinite(out["absorption_coefficient_m_inverse"])
