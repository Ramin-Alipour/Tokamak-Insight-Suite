"""Kinetic ECE emissivity/absorption integration over arbitrary EVDF grids.

The resonance kernel is supplied explicitly by a validated physics backend.
This module performs the distribution-weighted kinetic integrals and
radiative-transfer solution; it does not invent a universal cyclotron kernel.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid

def kinetic_ece_coefficients(frequency_Hz, momentum_axes, evdf,
                             emissivity_kernel, absorption_kernel,
                             *, path_length_m=None, background_intensity=None):
    """Integrate j_nu and alpha_nu for a supplied relativistic resonance kernel.

    Kernels have shape (Nfrequency,Np_parallel,Np_perp), and include all
    harmonic, polarization, Bessel, Doppler, and unit-normalization factors.
    EVDF is a non-negative phase-space density on (p_parallel,p_perp), with
    p_perp >= 0. The absorption kernel must encode the appropriate momentum
    derivative/induced-emission term for the selected convention.
    """
    f=np.asarray(frequency_Hz,float); pp,pt=[np.asarray(x,float) for x in momentum_axes]
    F=np.asarray(evdf,float); J=np.asarray(emissivity_kernel,float); A=np.asarray(absorption_kernel,float)
    if f.ndim!=1 or len(f)<1 or np.any(~np.isfinite(f)) or np.any(f<=0) or np.any(np.diff(f)<=0):
        raise ValueError("frequency_Hz must be finite, positive, and strictly increasing")
    if pp.ndim!=1 or pt.ndim!=1 or len(pp)<2 or len(pt)<2 or np.any(np.diff(pp)<=0) or np.any(pt<0) or np.any(np.diff(pt)<=0):
        raise ValueError("momentum axes must be increasing; p_perp must be non-negative")
    if F.shape!=(len(pp),len(pt)) or np.any(~np.isfinite(F)) or np.any(F<0):
        raise ValueError("EVDF must be finite, non-negative, and match momentum axes")
    expected=(len(f),len(pp),len(pt))
    if J.shape!=expected or A.shape!=expected or np.any(~np.isfinite(J)) or np.any(~np.isfinite(A)):
        raise ValueError("both kinetic kernels must be finite and have shape (frequency,p_parallel,p_perp)")
    # Cylindrical momentum-space Jacobian.  A common 2*pi factor belongs in
    # the supplied kernel normalization and must be documented by the caller.
    weighted=F[None,:,:]*pt[None,None,:]
    j=trapezoid(trapezoid(J*weighted,pt,axis=2),pp,axis=1)
    alpha=trapezoid(trapezoid(A*weighted,pt,axis=2),pp,axis=1)
    result={"frequency_Hz":f,"emissivity":j,"absorption_coefficient_m_inv":alpha,
            "kernel_contract":"caller-supplied relativistic harmonic/polarization kernel",
            "evdf_integrated":True,"radiative_transfer_solved":False}
    if path_length_m is not None:
        L=float(path_length_m)
        if not np.isfinite(L) or L<0: raise ValueError("path_length_m must be finite and non-negative")
        I0=np.zeros_like(f) if background_intensity is None else np.asarray(background_intensity,float)
        if I0.shape!=f.shape or np.any(~np.isfinite(I0)) or np.any(I0<0):
            raise ValueError("background_intensity must be finite, non-negative, and match frequency")
        tau=alpha*L
        # Stable formal solution including the alpha -> 0 limit.
        source=np.divide(j,alpha,out=np.zeros_like(j),where=np.abs(alpha)>1e-30)
        I=I0*np.exp(-tau)+source*(-np.expm1(-tau))
        I=np.where(np.abs(alpha)<=1e-30,I0+j*L,I)
        result.update(optical_depth=tau,emergent_intensity=I,radiative_transfer_solved=True)
    return result
