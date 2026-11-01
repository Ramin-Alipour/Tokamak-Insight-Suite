"""PSF-aware Thomson spatial-profile inverse for calibrated channel measurements.

The supplied spatial mixing matrix is a fixed, measured forward operator. TIS
never estimates or calibrates the instrument PSF. This is a regularized linear
reconstruction of one already-inferred physical parameter, not a joint
wavelength-resolved spectral inverse.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear

def reconstruct_profile_with_spatial_mixing(measured_channel_values, spatial_mixing_matrix, radius_m, *,
        uncertainty=None, covariance=None, parameter_name="physical_parameter", parameter_unit="",
        positive=False, regularization_strength=None, gcv_grid_size=81):
    """Invert supplied channel mixing y=H x with curvature regularization.

    Inputs are calibrated channel values and a fixed researcher-supplied spatial
    response operator. No PSF estimation, instrument calibration, or spectral fit
    is performed here.
    """
    y=np.asarray(measured_channel_values,dtype=float)
    h=np.asarray(spatial_mixing_matrix,dtype=float)
    x=np.asarray(radius_m,dtype=float)
    if y.ndim!=1 or y.size<3 or not np.all(np.isfinite(y)):
        raise ValueError("measured_channel_values must be a finite vector with at least three channels")
    if x.ndim!=1 or x.size<3 or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0):
        raise ValueError("radius_m must be finite, strictly increasing, and contain at least three nodes")
    if h.ndim!=2 or h.shape!=(y.size,x.size) or not np.all(np.isfinite(h)):
        raise ValueError("spatial_mixing_matrix must be finite with shape (n_channels, n_profile_nodes)")
    if np.linalg.matrix_rank(h)<min(h.shape):
        raise ValueError("spatial_mixing_matrix is rank deficient; supplied channel geometry cannot support this inverse")
    if uncertainty is not None and covariance is not None:
        raise ValueError("supply uncertainty or covariance, not both")
    if covariance is not None:
        c=np.asarray(covariance,dtype=float)
        if c.shape!=(y.size,y.size) or not np.all(np.isfinite(c)) or not np.allclose(c,c.T,rtol=1e-10,atol=1e-14):
            raise ValueError("covariance must be finite, symmetric, and match the channel count")
        try: chol=np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc: raise ValueError("covariance must be positive definite") from exc
        a=np.linalg.solve(chol,h); b=np.linalg.solve(chol,y)
        covariance_model="researcher_supplied_full_channel_covariance"
    elif uncertainty is not None:
        s=np.asarray(uncertainty,dtype=float)
        if s.shape!=y.shape or not np.all(np.isfinite(s)) or np.any(s<=0):
            raise ValueError("uncertainty must be positive, finite, and match measured_channel_values")
        a=h/s[:,None]; b=y/s
        covariance_model="independent_pointwise_channel_uncertainty"
    else:
        a=h.copy(); b=y.copy()
        covariance_model="unweighted_residual_scaled_covariance"
    d2=np.zeros((x.size-2,x.size),dtype=float)
    for i in range(x.size-2):
        left=x[i+1]-x[i]; right=x[i+2]-x[i+1]
        d2[i,i]=2/(left*(left+right))
        d2[i,i+1]=-2/(left*right)
        d2[i,i+2]=2/(right*(left+right))
    gram=a.T@a; rhs=a.T@b; penalty=d2.T@d2
    if regularization_strength is None:
        if isinstance(gcv_grid_size,(bool,np.bool_)) or not isinstance(gcv_grid_size,(int,np.integer)) or gcv_grid_size<21:
            raise ValueError("gcv_grid_size must be an integer >= 21")
        scale=max(float(np.trace(gram)/x.size),np.finfo(float).tiny)
        best=None
        for lam0 in scale*np.logspace(-10,8,int(gcv_grid_size)):
            system=gram+lam0*penalty
            try:
                candidate=np.linalg.solve(system,rhs)
                smoother=a@np.linalg.solve(system,a.T)
            except np.linalg.LinAlgError: continue
            resid=b-a@candidate; denom=1-float(np.trace(smoother))/y.size
            if denom<=1e-10: continue
            score=float(resid@resid/y.size/denom**2)
            if np.isfinite(score) and (best is None or score<best[0]):
                best=(score,float(lam0),candidate,smoother)
        if best is None: raise ValueError("GCV could not select a stable regularization strength for this geometry")
        gcv_score,lam,solution,smoother=best
        selection="generalized_cross_validation"
    else:
        if isinstance(regularization_strength,(bool,np.bool_)) or not np.isfinite(regularization_strength) or regularization_strength<0:
            raise ValueError("regularization_strength must be a finite non-negative scalar")
        lam=float(regularization_strength); system=gram+lam*penalty
        try:
            solution=np.linalg.solve(system,rhs)
            smoother=a@np.linalg.solve(system,a.T)
        except np.linalg.LinAlgError as exc: raise ValueError("regularized spatial inverse is singular") from exc
        resid=b-a@solution; denom=1-float(np.trace(smoother))/y.size
        gcv_score=float(resid@resid/y.size/denom**2) if denom>1e-10 else None
        selection="researcher_supplied"
    active=np.zeros(x.size,dtype=bool)
    if positive:
        aa=np.vstack((a,np.sqrt(lam)*d2)); bb=np.r_[b,np.zeros(d2.shape[0])]
        fit=lsq_linear(aa,bb,bounds=(0,np.inf),method="trf",lsmr_tol="auto")
        if not fit.success: raise ValueError(f"non-negative spatial inverse failed: {fit.message}")
        solution=fit.x; active=solution<=max(1e-12,1e-10*float(np.max(solution)))
    resid=b-a@solution; dof=float(np.trace(smoother)); system=gram+lam*penalty
    try: inv=np.linalg.inv(system)
    except np.linalg.LinAlgError as exc: raise ValueError("profile covariance unavailable: regularized Hessian is singular") from exc
    pcov=inv@gram@inv.T
    if covariance is None and uncertainty is None:
        pcov*=float(resid@resid/max(1,y.size-dof))
    if positive and np.any(active):
        free=~active; conditional=np.zeros_like(pcov)
        if np.any(free):
            reduced=system[np.ix_(free,free)]
            conditional[np.ix_(free,free)]=np.linalg.solve(reduced,gram[np.ix_(free,free)])@np.linalg.inv(reduced).T
        pcov=conditional
    return {
        "analysis_status":"completed","parameter_name":str(parameter_name),"parameter_unit":str(parameter_unit),
        "radius_m":x.copy(),"profile":solution,"profile_uncertainty":np.sqrt(np.maximum(np.diag(pcov),0)),
        "profile_covariance":pcov,"measured_channel_values":y.copy(),"fitted_channel_values":h@solution,
        "residual_whitened":resid,"spatial_mixing_operator":"researcher_supplied_fixed_forward_operator",
        "spatial_mixing_matrix":h.copy(),
        "regularization":{"kind":"nonuniform_grid_second_derivative_curvature_penalty","strength":lam,
            "selection":selection,"gcv_score":gcv_score,"effective_degrees_of_freedom":dof},
        "covariance_model":covariance_model,"jacobian_rank":int(np.linalg.matrix_rank(a)),
        "jacobian_condition_number":float(np.linalg.cond(a)),"active_nonnegative_nodes":np.flatnonzero(active).tolist(),
        "uncertainty_semantics":("local conditional covariance; for non-negative fits conditional on the active set"
            if positive else "linearized covariance conditional on supplied mixing operator and selected regularization"),
        "identifiability":{"primary":["regularized profile within supplied finite-dimensional spatial basis"],
            "conditional":["physical units/meaning are inherited from the upstream inferred parameter"],
            "non_identifiable":["spatial structures in the null space of the supplied mixing operator",
                "features finer than spatial resolution encoded by the supplied mixing operator"]},
        "experimental_validation":"not_performed; software/numerical inference only"
    }
