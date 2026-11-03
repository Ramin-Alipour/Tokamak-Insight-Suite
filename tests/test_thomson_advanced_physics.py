import numpy as np
from tokamak_analysis.cts_landau_fdt import landau_test_particle_fokker_planck
from tokamak_analysis.ece_kinetic import kinetic_ece_coefficients
from tokamak_analysis.cts_full_vector_em import propagate_full_vector_em

def test_landau_operator_shapes_and_psd():
    v=np.linspace(-3,3,7)
    X,Y,Z=np.meshgrid(v,v,v,indexing='ij')
    f=np.exp(-(X*X+Y*Y+Z*Z))
    out=landau_test_particle_fokker_planck(f,(v,v,v))
    assert out['collision_operator'].shape==f.shape
    assert out['diffusion_tensor'].shape==(3,3)+f.shape
    assert np.min(np.linalg.eigvalsh(np.moveaxis(out['diffusion_tensor'],(0,1),(-2,-1)))) > -1e-8
    assert 'not implemented' in out['fdt_scope']

def test_ece_kernel_integration_and_transfer():
    nu=np.array([1e9,2e9]); pp=np.linspace(-1,1,5); pt=np.linspace(0,1,4)
    F=np.ones((5,4)); K=np.ones((2,5,4))
    out=kinetic_ece_coefficients(nu,(pp,pt),F,K,K,path_length_m=0.2)
    assert out['radiative_transfer_solved']
    assert np.all(out['emergent_intensity']>=0)
    assert np.allclose(out['optical_depth'],out['absorption_coefficient_m_inv']*0.2)

def test_full_vector_vacuum_propagation():
    z=np.linspace(0,0.01,5)
    eps=lambda _z: np.eye(3,dtype=complex)
    y0=np.array([1+0j,0j,0j,1/376.730313668+0j])
    out=propagate_full_vector_em(z,eps,1e9,initial_state=y0,samples=10)
    assert out['state_Ex_Ey_Hx_Hy'].shape==(10,4)
    assert np.all(np.isfinite(out['state_Ex_Ey_Hx_Hy']))
