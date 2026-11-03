"""Full-vector, non-paraxial Maxwell propagation in a stratified anisotropic medium."""
from __future__ import annotations
import numpy as np
from scipy.constants import epsilon_0, mu_0, c
from scipy.integrate import solve_ivp

def propagate_full_vector_em(z_m, dielectric_tensor, frequency_Hz, *,
                             kx_rad_m=0.0, ky_rad_m=0.0, initial_state=None,
                             rtol=1e-8, atol=1e-10, samples=400):
    """Integrate the 1-D stratified Maxwell system for [Ex,Ey,Hx,Hy].

    dielectric_tensor(z) returns a complex 3x3 relative permittivity tensor.
    Arbitrary oblique incidence and anisotropy are retained; no paraxial
    expansion is made. This is a boundary-value initial-value propagator,
    not by itself a two-sided scattering-matrix/mode-conversion solver.
    """
    z=np.asarray(z_m,float); w=2*np.pi*float(frequency_Hz); kx=float(kx_rad_m); ky=float(ky_rad_m)
    if z.ndim!=1 or len(z)<2 or np.any(~np.isfinite(z)) or np.any(np.diff(z)<=0):
        raise ValueError("z_m must be finite and strictly increasing")
    if not np.isfinite(w) or w<=0 or not np.isfinite(kx+ky): raise ValueError("invalid frequency or transverse wavevector")
    def eps_at(zz):
        e=np.asarray(dielectric_tensor(float(zz)),complex)
        if e.shape!=(3,3) or np.any(~np.isfinite(e)): raise ValueError("dielectric_tensor(z) must return a finite complex 3x3 tensor")
        if abs(e[2,2])<1e-14: raise ValueError("epsilon_zz is singular; this coordinate formulation is ill-conditioned")
        return e
    if initial_state is None: raise ValueError("initial_state [Ex,Ey,Hx,Hy] is required")
    y0=np.asarray(initial_state,complex)
    if y0.shape!=(4,) or np.any(~np.isfinite(y0)): raise ValueError("initial_state must be four finite complex components")
    def rhs(zz,y):
        ex,ey,hx,hy=y; e=eps_at(zz)
        ez=(-(kx*hy-ky*hx)/(w*epsilon_0)-e[2,0]*ex-e[2,1]*ey)/e[2,2]
        hz=(kx*ey-ky*ex)/(w*mu_0)
        dEx=1j*kx*ez+1j*w*mu_0*hy
        dEy=1j*ky*ez-1j*w*mu_0*hx
        dHx=1j*ky*hz-1j*w*epsilon_0*(e[1,0]*ex+e[1,1]*ey+e[1,2]*ez)
        dHy=1j*w*epsilon_0*(e[0,0]*ex+e[0,1]*ey+e[0,2]*ez)-1j*kx*hz
        return np.array([dEx,dEy,dHx,dHy])
    zz=np.linspace(z[0],z[-1],int(samples))
    if len(zz)<2: raise ValueError("samples must be >=2")
    sol=solve_ivp(rhs,(z[0],z[-1]),y0,t_eval=zz,rtol=rtol,atol=atol,method="DOP853")
    if not sol.success: raise RuntimeError("Maxwell propagation failed: "+sol.message)
    return {"z_m":sol.t,"state_Ex_Ey_Hx_Hy":sol.y.T,
            "frequency_Hz":float(frequency_Hz),"kx_rad_m":kx,"ky_rad_m":ky,
            "method":"full-vector non-paraxial 1-D stratified Maxwell ODE",
            "limitations":["Requires a stratified medium (epsilon depends on z only).",
                           "Initial-value propagation only; reflection and complete mode-conversion coefficients require two-sided boundary matching.",
                           "No automatic cold-plasma dielectric tensor is assumed."]}
