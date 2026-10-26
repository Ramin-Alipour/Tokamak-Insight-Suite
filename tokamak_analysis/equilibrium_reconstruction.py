"""Finite-difference Grad-Shafranov equilibrium solver for supplied profiles and boundary flux."""
from __future__ import annotations
import numpy as np
MU0 = 4e-7 * np.pi

def reconstruct_equilibrium(R_m, Z_m, boundary_flux_Wb_rad, psi_profile, pressure_gradient_Pa_per_Wb, ffprime_T2, *, max_iterations=10000, tolerance=1e-8, relaxation=1.2):
    """Solve axisymmetric Grad-Shafranov equation on a uniform R-Z grid.

    Boundary flux is fixed on the grid perimeter. Profile arrays are interpolated
    against normalized flux estimated from the current iterate. This is a numerical
    solver, not a complete experimental EFIT replacement.
    """
    R = np.asarray(R_m, dtype=float); Z = np.asarray(Z_m, dtype=float)
    bnd = np.asarray(boundary_flux_Wb_rad, dtype=float)
    prof = np.asarray(psi_profile, dtype=float)
    pp = np.asarray(pressure_gradient_Pa_per_Wb, dtype=float)
    ff = np.asarray(ffprime_T2, dtype=float)
    if R.ndim != 1 or Z.ndim != 1 or len(R)<3 or len(Z)<3: raise ValueError("R and Z need at least 3 grid points")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or np.any(R<=0) or np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0): raise ValueError("R must be positive and both grids strictly increasing and finite")
    if not np.allclose(np.diff(R), np.diff(R)[0]) or not np.allclose(np.diff(Z), np.diff(Z)[0]): raise ValueError("Uniform grids are required")
    if bnd.shape != (len(Z),len(R)) or not np.all(np.isfinite(bnd)): raise ValueError("Boundary flux must be finite with shape (len(Z), len(R))")
    if prof.ndim != 1 or len(prof)<2 or pp.shape != prof.shape or ff.shape != prof.shape or not np.all(np.isfinite(prof)) or not np.all(np.isfinite(pp)) or not np.all(np.isfinite(ff)): raise ValueError("Profile arrays must be finite 1-D arrays of equal length")
    if np.any(np.diff(prof)<=0) or prof[0]<0 or prof[-1]>1: raise ValueError("psi_profile must increase strictly within [0,1]")
    if not np.isfinite(max_iterations) or int(max_iterations)!=max_iterations or max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(relaxation) or not 0<relaxation<2: raise ValueError("Solver controls require a positive integer iteration limit, finite positive tolerance, and finite relaxation in (0,2)")
    psi = np.zeros_like(bnd); psi[0,:]=bnd[0,:]; psi[-1,:]=bnd[-1,:]; psi[:,0]=bnd[:,0]; psi[:,-1]=bnd[:,-1]
    dr=R[1]-R[0]; dz=Z[1]-Z[0]; ar=1/dr**2; az=1/dz**2; converged=False; change=np.inf
    for it in range(1,int(max_iterations)+1):
        old=psi.copy(); lo=float(np.min(psi)); hi=float(np.max(psi)); span=hi-lo
        psin=np.zeros_like(psi) if span<=np.finfo(float).eps*max(1,abs(lo),abs(hi)) else (psi-lo)/span
        source=-MU0*R[None,:]**2*np.interp(psin,prof,pp,left=pp[0],right=pp[-1])-np.interp(psin,prof,ff,left=ff[0],right=ff[-1])
        for j in range(1,len(Z)-1):
            for i in range(1,len(R)-1):
                rp=ar+1/(2*R[i]*dr); rm=ar-1/(2*R[i]*dr); den=-2*ar-2*az
                candidate=(source[j,i]-rp*psi[j,i+1]-rm*psi[j,i-1]-az*(psi[j+1,i]+psi[j-1,i]))/den
                psi[j,i]=(1-relaxation)*psi[j,i]+relaxation*candidate
        change=float(np.max(np.abs(psi-old)))/max(float(np.max(np.abs(psi))),np.finfo(float).tiny)
        if change<tolerance: converged=True; break
    dpsi_dz,dpsi_dr=np.gradient(psi,Z,R,edge_order=2)
    br=-dpsi_dz/R[None,:]; bz=dpsi_dr/R[None,:]
    span=float(np.max(psi)-np.min(psi)); psin=np.zeros_like(psi) if span==0 else (psi-np.min(psi))/span
    return {"R_m":R.copy(),"Z_m":Z.copy(),"psi_Wb_rad":psi,"psi_normalized":psin,"B_R_T":br,"B_Z_T":bz,"B_poloidal_T":np.hypot(br,bz),"iterations":it,"relative_iteration_change":change,"converged":converged,"warning":"Check nonlinear residual, boundary constraints, profile conventions, and experimental magnetic constraints before scientific interpretation."}

def grad_shafranov_physical_relation(R_m, Z_m, psi_Wb_rad, pressure_gradient_Pa_per_Wb, ffprime_T2):
    """Evaluate the Grad-Shafranov equation Δ*ψ = -μ0 R² p'(ψ) - F F'(ψ).

    Returns the Grad-Shafranov operator, physical source term, and equation
    residual on a measured/reconstructed uniform or nonuniform R-Z grid.
    The supplied profile terms must already be mapped onto the grid.
    """
    R = np.asarray(R_m, dtype=float)
    Z = np.asarray(Z_m, dtype=float)
    psi = np.asarray(psi_Wb_rad, dtype=float)
    dp = np.asarray(pressure_gradient_Pa_per_Wb, dtype=float)
    ff = np.asarray(ffprime_T2, dtype=float)
    expected = (Z.size, R.size)
    if R.ndim != 1 or Z.ndim != 1 or R.size < 3 or Z.size < 3:
        raise ValueError("R and Z must be 1-D grids with at least three points")
    if np.any(R <= 0) or np.any(np.diff(R) <= 0) or np.any(np.diff(Z) <= 0):
        raise ValueError("R must be positive and both grids strictly increasing")
    if any(a.shape != expected for a in (psi, dp, ff)):
        raise ValueError("psi, pressure-gradient, and FF-prime arrays must have shape (len(Z), len(R))")
    if not all(np.all(np.isfinite(a)) for a in (R, Z, psi, dp, ff)):
        raise ValueError("All inputs must be finite")
    dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
    d2psi_dz2 = np.gradient(dpsi_dz, Z, axis=0, edge_order=2)
    d2psi_dr2 = np.gradient(dpsi_dr, R, axis=1, edge_order=2)
    operator = d2psi_dr2 - dpsi_dr / R[None, :] + d2psi_dz2
    source = -MU0 * R[None, :]**2 * dp - ff
    return {
        "grad_shafranov_operator_Wb_rad_m-2": operator,
        "physical_source_Wb_rad_m-2": source,
        "equation_residual_Wb_rad_m-2": operator - source,
        "equation": "Δ*ψ = -μ0 R² dp/dψ - F dF/dψ",
    }
