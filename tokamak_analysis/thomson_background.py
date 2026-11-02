"""Structured detector-space background models for calibrated Thomson spectra."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline


def fit_structured_background(wavelength_nm, spectrum, *, model="linear",
                              degree=2, interior_knots=3, covariance=None):
    x, y = np.asarray(wavelength_nm), np.asarray(spectrum)
    if x.dtype.kind not in "iuf" or y.dtype.kind not in "iuf":
        raise ValueError("wavelength and spectrum must be real numeric arrays")
    x, y = x.astype(float), y.astype(float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 5 or not np.all(np.isfinite(x+y)) or np.any(np.diff(x)<=0):
        raise ValueError("wavelength and spectrum must be matching finite vectors on an increasing grid")
    z = 2*(x-x.min())/np.ptp(x)-1
    if model == "constant":
        design = np.ones((x.size,1))
    elif model == "linear":
        design = np.column_stack((np.ones(x.size), z))
    elif model == "polynomial":
        if isinstance(degree, bool) or not isinstance(degree, (int,np.integer)) or not 0 <= degree <= 8:
            raise ValueError("degree must be an integer from 0 to 8")
        design = np.polynomial.legendre.legvander(z, int(degree))
    elif model == "cubic_spline":
        if isinstance(interior_knots,bool) or not isinstance(interior_knots,(int,np.integer)) or interior_knots < 0:
            raise ValueError("interior_knots must be a non-negative integer")
        inner=np.linspace(x.min(),x.max(),int(interior_knots)+2)[1:-1]
        knots=np.r_[np.repeat(x.min(),4),inner,np.repeat(x.max(),4)]
        design=BSpline.design_matrix(x,knots,3,extrapolate=False).toarray()
    else:
        raise ValueError("model must be constant, linear, polynomial or cubic_spline")
    if covariance is None:
        a,b=design,y
        noise="unweighted_residual_scaled"
    else:
        c=np.asarray(covariance)
        if c.dtype.kind not in "iuf" or c.shape!=(x.size,x.size) or not np.all(np.isfinite(c)) or not np.allclose(c,c.T,rtol=1e-10,atol=1e-14):
            raise ValueError("covariance must be finite, symmetric and match wavelength grid")
        try: chol=np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc: raise ValueError("covariance must be positive definite") from exc
        a,b=np.linalg.solve(chol,design),np.linalg.solve(chol,y)
        noise="full_spectral_covariance"
    coef,_,rank,s=np.linalg.lstsq(a,b,rcond=None)
    if rank < design.shape[1]:
        raise ValueError("background basis is not identifiable")
    residual=y-design@coef
    dof=max(1,x.size-design.shape[1])
    scale=float(residual@residual/dof) if covariance is None else 1.0
    return {
        "background":design@coef,"coefficients":coef,
        "coefficient_covariance":np.linalg.pinv(a.T@a)*scale,
        "residual":residual,"basis_matrix":design,"model":model,
        "noise_model":noise,"fit_rank":int(rank),
        "condition_number":float(s[0]/s[-1]),"degrees_of_freedom":int(dof),
        "background_role":"detector_space_nuisance_not_plasma_emission",
        "limitations":["No background-only wavelength mask is inferred.",
                       "Use as a nuisance component in a joint physical fit when background overlaps the Thomson feature.",
                       "Spline complexity is user-selected and is not evidence for a physical continuum."]
    }
