"""Velocity-space Landau collision operator on a uniform 3-D grid.

The deterministic operator is the Rosenbluth-potential (test-particle
Fokker--Planck) form.  The returned diffusion covariance is the local
Langevin/test-particle fluctuation covariance.  It is NOT the complete
collective Boltzmann--Langevin FDT operator: field-particle correlations and
a conservative discrete variational formulation remain explicitly outside
this implementation.
"""
from __future__ import annotations
import numpy as np
from scipy.constants import elementary_charge as E_CHARGE
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

def landau_test_particle_fokker_planck(f, velocity_axes, *, coupling=1.0,
                                       softening=None):
    """Evaluate C[f]=d_i(D_ij d_j f - F_i f) on a Cartesian velocity grid.

    Parameters
    ----------
    f : (Nv,Nv,Nv) non-negative distribution.
    velocity_axes : three equally spaced, strictly increasing 1-D arrays [m/s].
    coupling : non-negative Landau prefactor (units must match f and v).
    softening : positive velocity-space softening [m/s]. Defaults to half a
        grid cell. A finite softening regularizes the integrable grid singularity.

    Returns D_ij, F_i, collision_operator, and local_noise_covariance_ij.
    The noise covariance is 2 D_ij in the Ito velocity SDE convention;
    multiply by coupling and the particle normalization used by the caller.
    """
    a=[np.asarray(x,float) for x in velocity_axes]
    q=np.asarray(f,float)
    if q.ndim!=3 or any(x.ndim!=1 or len(x)<4 or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0) for x in a):
        raise ValueError("f and three finite, increasing velocity axes (>=4 points) are required")
    steps=[np.diff(x) for x in a]
    if any(not np.allclose(s,s[0],rtol=1e-9,atol=0) for s in steps):
        raise ValueError("velocity axes must be uniformly spaced")
    if q.shape!=tuple(len(x) for x in a) or np.any(~np.isfinite(q)) or np.any(q<0):
        raise ValueError("f must match the grid and be finite and non-negative")
    if not np.isfinite(coupling) or coupling<0:
        raise ValueError("coupling must be finite and non-negative")
    dv=[s[0] for s in steps]; cell=float(np.prod(dv))
    h=float(softening) if softening is not None else 0.5*min(dv)
    if not np.isfinite(h) or h<=0: raise ValueError("softening must be positive")
    coords=np.meshgrid(*[np.arange(-(n-1),n)*d for n,d in zip(q.shape,dv)],indexing="ij")
    r2=sum(c*c for c in coords)+h*h
    # U_ij=(delta_ij/r-r_i*r_j/r^3), and d_j U_ij=-2 r_i/r^3.
    U=np.empty((3,3)+r2.shape); G=np.empty((3,)+r2.shape)
    r=np.sqrt(r2)
    for i in range(3):
        G[i]=-2*coords[i]/(r2*r)
        for j in range(3):
            U[i,j]=((1.0/r) if i==j else 0.0)-coords[i]*coords[j]/(r2*r)
    grad=np.stack(np.gradient(q,*dv,edge_order=2))
    D=np.empty((3,3)+q.shape); drift=np.empty((3,)+q.shape)
    for i in range(3):
        drift[i]=coupling*cell*fftconvolve(q,G[i],mode="same")
        for j in range(3):
            D[i,j]=coupling*cell*fftconvolve(q,U[i,j],mode="same")
    flux=np.einsum("ij...,j...->i...",D,grad)-drift*q
    collision=sum(np.gradient(flux[i],dv[i],axis=i,edge_order=2) for i in range(3))
    D=0.5*(D+np.swapaxes(D,0,1))
    eig=np.linalg.eigvalsh(np.moveaxis(D,(0,1),(-2,-1)))
    if np.min(eig)<-1e-8*max(1.0,float(np.max(np.abs(D)))):
        raise FloatingPointError("Landau diffusion tensor lost positive semidefiniteness")
    return {"collision_operator":collision,"diffusion_tensor":D,
            "friction_vector":drift,"local_noise_covariance":2.0*D,
            "operator":"3-D Landau test-particle Fokker-Planck",
            "fdt_scope":"local Ito test-particle covariance only; collective field-particle FDT not implemented",
            "grid_cell_volume":cell,"softening_m_s":h}
