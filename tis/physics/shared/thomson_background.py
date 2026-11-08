"""Structured detector-space background models for calibrated Thomson spectra.

Background terms are nuisance components in detector space.  They are not
interpreted as plasma emission and should be fitted jointly with a physical
Thomson forward model whenever the background overlaps the Thomson feature.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _real_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    out = raw.astype(float, copy=False)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _positive_definite_covariance(value, size, name="covariance"):
    c = _real_array(value, name, 2)
    if c.shape != (size, size) or not np.allclose(c, c.T, rtol=1e-10, atol=1e-14):
        raise ValueError(f"{name} must be symmetric and match the flattened spectra")
    c = 0.5 * (c + c.T)
    try:
        return c, np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def _background_design(wavelength_nm, model, *, degree, interior_knots, templates):
    x = wavelength_nm
    z = 2.0 * (x - x.min()) / np.ptp(x) - 1.0
    if model == "constant":
        design = np.ones((x.size, 1))
    elif model == "linear":
        design = np.column_stack((np.ones(x.size), z))
    elif model == "polynomial":
        if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or not 0 <= degree <= 8:
            raise ValueError("degree must be an integer from 0 to 8")
        design = np.polynomial.legendre.legvander(z, int(degree))
    elif model == "cubic_spline":
        if isinstance(interior_knots, (bool, np.bool_)) or not isinstance(interior_knots, (int, np.integer)) or interior_knots < 0:
            raise ValueError("interior_knots must be a non-negative integer")
        inner = np.linspace(x.min(), x.max(), int(interior_knots) + 2)[1:-1]
        knots = np.r_[np.repeat(x.min(), 4), inner, np.repeat(x.max(), 4)]
        design = BSpline.design_matrix(x, knots, 3, extrapolate=False).toarray()
    elif model == "templates":
        if templates is None:
            raise ValueError("templates model requires a measured/researcher-supplied template matrix")
        design = _real_array(templates, "templates", 2)
        if design.shape[0] != x.size or design.shape[1] < 1:
            raise ValueError("templates must have one row per wavelength sample and at least one column")
    else:
        raise ValueError("model must be constant, linear, polynomial, cubic_spline or templates")
    if not np.all(np.isfinite(design)) or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("background basis must be finite and full column rank")
    return design


def fit_structured_background(wavelength_nm, spectrum, *, model="linear",
                              degree=2, interior_knots=3, covariance=None,
                              templates=None):
    """Fit a selected detector-space background basis using GLS or OLS."""
    x = _real_array(wavelength_nm, "wavelength_nm", 1)
    y = _real_array(spectrum, "spectrum", 1)
    if y.shape != x.shape or x.size < 5 or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength and spectrum must be matching finite vectors on an increasing grid")
    design = _background_design(
        x, model, degree=degree, interior_knots=interior_knots, templates=templates
    )
    if covariance is None:
        a, b = design, y
        noise = "unweighted_residual_scaled"
    else:
        c, chol = _positive_definite_covariance(covariance, x.size)
        a, b = np.linalg.solve(chol, design), np.linalg.solve(chol, y)
        noise = "full_spectral_covariance"
    coef, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("background basis is not identifiable")
    residual = y - design @ coef
    dof = max(1, x.size - design.shape[1])
    scale = float(np.linalg.norm(np.linalg.solve(chol, residual))**2 / dof) if covariance is not None else float(residual @ residual / dof)
    # With a supplied covariance, GLS already contains the absolute noise
    # scale: Cov(beta) = (X.T C^-1 X)^-1.  Multiplying by reduced chi-square
    # would double-scale known measurement errors and understate/overstate
    # uncertainty whenever the fitted model is imperfect.  Only the
    # unweighted route estimates a residual variance from the data.
    coefficient_covariance = np.linalg.pinv(a.T @ a) * (
        1.0 if covariance is not None else scale
    )
    return {
        "background": design @ coef, "coefficients": coef,
        "coefficient_covariance": coefficient_covariance,
        "residual": residual, "basis_matrix": design, "model": model,
        "noise_model": noise, "fit_rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "degrees_of_freedom": int(dof),
        "background_role": "detector_space_nuisance_not_plasma_emission",
        "limitations": [
            "No background-only wavelength mask is inferred.",
            "Use as a nuisance component in a joint physical fit when background overlaps the Thomson feature.",
            "Spline complexity and template choice are researcher-selected; neither establishes a physical continuum.",
            "A standalone background fit is not a substitute for simultaneous Thomson-signal/background inference.",
        ],
    }


def fit_multichannel_structured_background(
    wavelength_nm, spectra, *, model="linear", degree=2, interior_knots=3,
    covariance=None, templates=None,
):
    """Jointly fit per-channel background coefficients with full cross-channel GLS.

    The covariance is flattened in channel-major order.  A shared template
    basis may be supplied, but each channel has its own coefficients.  This
    estimates detector-space nuisance terms only; it does not fit Thomson
    physics or estimate a spatial PSF.
    """
    x = _real_array(wavelength_nm, "wavelength_nm")
    y = _real_array(spectra, "spectra", 2)
    if x.ndim == 1:
        if x.size != y.shape[1]:
            raise ValueError("shared wavelength axis must match the spectral width")
        axes = np.broadcast_to(x, y.shape)
    elif x.shape == y.shape:
        axes = x
    else:
        raise ValueError("wavelength_nm must be shared 1-D or match spectra shape")
    if y.shape[0] < 1 or y.shape[1] < 5 or np.any(axes <= 0) or np.any(np.diff(axes, axis=1) <= 0):
        raise ValueError("spectra need at least one channel and five samples on increasing positive wavelength axes")
    designs = [
        _background_design(
            axes[i], model, degree=degree, interior_knots=interior_knots,
            templates=(templates[i] if isinstance(templates, (list, tuple)) else templates),
        )
        for i in range(y.shape[0])
    ]
    nchan, nlam = y.shape
    ncoef = designs[0].shape[1]
    if any(d.shape[1] != ncoef for d in designs):
        raise ValueError("all channel background bases must have the same number of columns")
    design = np.zeros((nchan * nlam, nchan * ncoef), dtype=float)
    for i, d in enumerate(designs):
        design[i*nlam:(i+1)*nlam, i*ncoef:(i+1)*ncoef] = d
    observed = y.ravel()
    if covariance is None:
        a, b, noise = design, observed, "unweighted_residual_scaled"
    else:
        c, chol = _positive_definite_covariance(covariance, observed.size, "joint_covariance")
        a, b, noise = np.linalg.solve(chol, design), np.linalg.solve(chol, observed), "full_channel_major_spectral_covariance"
    coefficients, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("joint channel background basis is not identifiable")
    fitted = (design @ coefficients).reshape(y.shape)
    residual = y - fitted
    dof = max(1, observed.size - design.shape[1])
    whitened_residual = np.linalg.solve(chol, residual.ravel()) if covariance is not None else residual.ravel()
    scale = float(whitened_residual @ whitened_residual / dof)
    coefficient_covariance = np.linalg.pinv(a.T @ a) * (1.0 if covariance is not None else scale)
    return {
        "background": fitted, "coefficients": coefficients.reshape(nchan, ncoef),
        "coefficient_covariance": coefficient_covariance,
        "residual": residual, "design_matrix_channel_major": design,
        "basis_matrices": designs, "model": model, "noise_model": noise,
        "fit_rank": int(rank), "parameter_count": int(design.shape[1]),
        "condition_number": float(singular[0] / singular[-1]),
        "degrees_of_freedom": int(dof),
        "covariance_ordering": "channel_major",
        "cross_channel_covariance_used": covariance is not None,
        "background_role": "detector_space_nuisance_not_plasma_emission",
        "limitations": [
            "This is a background-only nuisance fit, not a joint physical Thomson inversion.",
            "Cross-channel covariance is retained in coefficient covariance when supplied.",
            "No channel mixing/PSF is estimated or applied.",
            "Template and spline choices remain researcher-supplied model assumptions.",
        ],
    }
