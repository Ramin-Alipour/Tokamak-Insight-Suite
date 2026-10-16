"""Regularized linear reconstruction of emissivity from line-integrated signals."""
import numpy as np
def reconstruct_emissivity(response_matrix, line_integrals, *, regularization=1e-6, nonnegative=True):
    A=np.asarray(response_matrix,dtype=float); y=np.asarray(line_integrals,dtype=float); lam=float(regularization)
    if A.ndim!=2 or y.ndim!=1 or A.shape[0]!=len(y) or min(A.shape)<1 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)): raise ValueError("Finite response matrix and matching measurements required")
    if not np.isfinite(lam) or lam<0: raise ValueError("Regularization must be nonnegative")
    lhs=A.T@A+lam*np.eye(A.shape[1]); rhs=A.T@y
    if nonnegative:
        from scipy.optimize import lsq_linear
        augA=np.vstack([A,np.sqrt(lam)*np.eye(A.shape[1])]) if lam>0 else A
        augy=np.r_[y,np.zeros(A.shape[1])] if lam>0 else y
        sol=lsq_linear(augA,augy,bounds=(0,np.inf)).x
    else: sol=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    residual=A@sol-y
    return {"emissivity":sol,"synthetic_measurements":A@sol,"residual":residual,"residual_norm":float(np.linalg.norm(residual)),"regularization":lam,"caveat":"Reconstruction quality depends on calibrated geometry matrix, coverage, regularization, and uncertainty model."}

def abel_project_axisymmetric_emissivity(radius_m, emissivity_W_m3, impact_parameter_m):
    """Forward Abel projection g(p)=2*integral_p^a epsilon(r)*r/sqrt(r^2-p^2) dr."""
    r=np.asarray(radius_m,dtype=float); eps=np.asarray(emissivity_W_m3,dtype=float); p=np.asarray(impact_parameter_m,dtype=float)
    if r.ndim!=1 or eps.shape!=r.shape or len(r)<3 or p.ndim!=1 or len(p)<1 or any(not np.all(np.isfinite(v)) for v in (r,eps,p)):
        raise ValueError("Radius/emissivity must match and all inputs be finite 1-D arrays")
    if np.any(np.diff(r)<=0) or not np.isclose(r[0],0.0,atol=1e-12) or np.any(eps<0) or np.any(p<0) or np.any(p>=r[-1]):
        raise ValueError("Require radius from zero, non-negative emissivity and impact parameters in [0,a)")
    values=[]
    for chord in p:
        mask=r>chord
        rr=r[mask]; ee=eps[mask]
        if len(rr)<2: values.append(0.0); continue
        integrand=ee*rr/np.sqrt(rr**2-chord**2)
        # Avoid the integrable endpoint singularity by omitting the first sample adjacent to p.
        values.append(float(2*np.trapezoid(integrand,rr)))
    return {"impact_parameter_m":p,"line_integrated_emissivity_W_m2":np.asarray(values),
            "equation":"g(p)=2*integral_p^a epsilon(r)*r/sqrt(r^2-p^2) dr",
            "caveat":"Discrete forward Abel projection; near-tangent chords require adequate radial resolution and endpoint treatment."}


def forward_project_emissivity(response_matrix, emissivity):
    """Forward tomography model y=A*epsilon, mapping voxel emissivity to line-integrated measurements."""
    A=np.asarray(response_matrix,dtype=float); eps=np.asarray(emissivity,dtype=float)
    if A.ndim!=2 or eps.ndim!=1 or A.shape[1]!=len(eps) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(eps)):
        raise ValueError("Finite 2-D response matrix and matching 1-D emissivity vector required")
    if np.any(eps<0): raise ValueError("Emissivity must be non-negative")
    return {"predicted_line_integrals":A@eps,"equation":"y=A*epsilon",
            "caveat":"Response matrix must encode calibrated line-of-sight geometry and detector response."}
