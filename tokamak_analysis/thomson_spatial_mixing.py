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
        positive=False, regularization_strength=None, gcv_grid_size=81,
        radius_covariance_m2=None, radius_mc_samples=200, radius_mc_seed=84621,
        spatial_mixing_covariance=None, mixing_mc_samples=200, mixing_mc_seed=84622):
    """Invert supplied channel mixing y=H x with curvature regularization.

    Inputs are calibrated channel values and a fixed researcher-supplied spatial
    response operator. No PSF estimation, instrument calibration, or spectral fit
    is performed here. Optional spatial_mixing_covariance is the covariance of
    the flattened matrix entries in row-major (C) order; it is propagated by
    Monte Carlo while holding the selected regularization strength fixed.
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
    coordinate_covariance = None
    if radius_covariance_m2 is not None:
        raw_coordinate_covariance = np.asarray(radius_covariance_m2)
        if raw_coordinate_covariance.dtype.kind not in "iuf":
            raise ValueError("radius_covariance_m2 must contain real numeric values")
        coordinate_covariance = raw_coordinate_covariance.astype(float, copy=False)
        if (coordinate_covariance.shape != (x.size, x.size)
                or not np.all(np.isfinite(coordinate_covariance))
                or not np.allclose(coordinate_covariance, coordinate_covariance.T,
                                   rtol=1e-10, atol=1e-18)):
            raise ValueError("radius_covariance_m2 must be finite, symmetric and match radius_m")
        coordinate_covariance = 0.5*(coordinate_covariance+coordinate_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(coordinate_covariance)
        coordinate_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if float(np.min(eigenvalues)) < -1e-10*coordinate_scale:
            raise ValueError("radius_covariance_m2 must be positive semidefinite")
        coordinate_covariance = (eigenvectors*np.maximum(eigenvalues, 0.0))@eigenvectors.T
        if (isinstance(radius_mc_samples, (bool, np.bool_))
                or not isinstance(radius_mc_samples, (int, np.integer))
                or radius_mc_samples < 20):
            raise ValueError("radius_mc_samples must be an integer >= 20 when coordinate covariance is supplied")
        if isinstance(radius_mc_seed, (bool, np.bool_)) or not isinstance(radius_mc_seed, (int, np.integer)):
            raise ValueError("radius_mc_seed must be an integer")
    elif radius_mc_samples != 200:
        raise ValueError("radius_mc_samples is meaningful only when radius_covariance_m2 is supplied")

    mixing_covariance = None
    if spatial_mixing_covariance is not None:
        raw_mixing_covariance = np.asarray(spatial_mixing_covariance)
        if raw_mixing_covariance.dtype.kind not in "iuf":
            raise ValueError("spatial_mixing_covariance must contain real numeric values")
        mixing_covariance = raw_mixing_covariance.astype(float, copy=False)
        if (mixing_covariance.shape != (h.size, h.size)
                or not np.all(np.isfinite(mixing_covariance))
                or not np.allclose(mixing_covariance, mixing_covariance.T,
                                   rtol=1e-10, atol=1e-14)):
            raise ValueError("spatial_mixing_covariance must be finite, symmetric, and match the flattened spatial_mixing_matrix")
        mixing_covariance = 0.5*(mixing_covariance+mixing_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(mixing_covariance)
        mixing_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if float(np.min(eigenvalues)) < -1e-10*mixing_scale:
            raise ValueError("spatial_mixing_covariance must be positive semidefinite")
        mixing_covariance = (eigenvectors*np.maximum(eigenvalues, 0.0))@eigenvectors.T
        if (isinstance(mixing_mc_samples, (bool,np.bool_))
                or not isinstance(mixing_mc_samples, (int,np.integer))
                or mixing_mc_samples < 20):
            raise ValueError("mixing_mc_samples must be an integer >= 20 when PSF covariance is supplied")
        if isinstance(mixing_mc_seed, (bool,np.bool_)) or not isinstance(mixing_mc_seed, (int,np.integer)):
            raise ValueError("mixing_mc_seed must be an integer")
    elif mixing_mc_samples != 200:
        raise ValueError("mixing_mc_samples is meaningful only when spatial_mixing_covariance is supplied")

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
    measurement_profile_covariance = pcov.copy()
    # Decide whether the nominal measured operator identifies every latent
    # profile node before propagating coordinate or PSF uncertainty. Random
    # perturbations must not turn a non-identifiable nominal inverse into a
    # reportable physical profile.
    coordinate_profile_covariance = None
    coordinate_successful_draws = 0
    coordinate_warning = None
    if coordinate_covariance is not None and data_identifiable:
        rng = np.random.default_rng(int(radius_mc_seed))
        eigenvalues, eigenvectors = np.linalg.eigh(coordinate_covariance)
        root = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.0))
        profile_draws = []
        # Hold the selected regularization strength fixed: this propagates
        # coordinate uncertainty, not GCV model-selection variability.
        for _ in range(int(radius_mc_samples)):
            sampled_radius = x + rng.standard_normal(x.size)@root.T
            if not np.all(np.isfinite(sampled_radius)) or np.any(np.diff(sampled_radius) <= 0):
                continue
            try:
                sampled = reconstruct_profile_with_spatial_mixing(
                    y, h, sampled_radius, uncertainty=uncertainty,
                    covariance=covariance, parameter_name=parameter_name,
                    parameter_unit=parameter_unit, positive=positive,
                    regularization_strength=lam, gcv_grid_size=gcv_grid_size,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if (sampled.get("primary_output_reportable") is True
                    and sampled.get("profile") is not None
                    and np.all(np.isfinite(sampled["profile"]))):
                profile_draws.append(np.asarray(sampled["profile"], dtype=float))
        coordinate_successful_draws = len(profile_draws)
        minimum_draws = max(20, int(np.ceil(0.8*int(radius_mc_samples))))
        if coordinate_successful_draws >= minimum_draws:
            coordinate_profile_covariance = np.cov(np.asarray(profile_draws), rowvar=False, ddof=1)
            pcov = measurement_profile_covariance + coordinate_profile_covariance
        else:
            coordinate_warning = (
                "Too few monotonic, numerically valid coordinate draws; reported profile covariance "
                "remains conditional on the nominal supplied radius_m."
            )
    mixing_profile_covariance = None
    mixing_successful_draws = 0
    mixing_warning = None
    if mixing_covariance is not None and data_identifiable:
        rng = np.random.default_rng(int(mixing_mc_seed))
        eigenvalues, eigenvectors = np.linalg.eigh(mixing_covariance)
        root = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.0))
        profile_draws = []
        for _ in range(int(mixing_mc_samples)):
            sampled_h = h + (rng.standard_normal(h.size)@root.T).reshape(h.shape)
            if not np.all(np.isfinite(sampled_h)) or np.any(sampled_h < 0.0):
                continue
            try:
                sampled = reconstruct_profile_with_spatial_mixing(
                    y, sampled_h, x, uncertainty=uncertainty, covariance=covariance,
                    parameter_name=parameter_name, parameter_unit=parameter_unit,
                    positive=positive, regularization_strength=lam,
                    gcv_grid_size=gcv_grid_size,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if (sampled.get("primary_output_reportable") is True
                    and sampled.get("profile") is not None
                    and np.all(np.isfinite(sampled["profile"]))):
                profile_draws.append(np.asarray(sampled["profile"], dtype=float))
        mixing_successful_draws = len(profile_draws)
        minimum_draws = max(20, int(np.ceil(0.8*int(mixing_mc_samples))))
        if mixing_successful_draws >= minimum_draws:
            mixing_profile_covariance = np.cov(np.asarray(profile_draws), rowvar=False, ddof=1)
            pcov = measurement_profile_covariance
            if coordinate_profile_covariance is not None:
                pcov = pcov + coordinate_profile_covariance
            pcov = pcov + mixing_profile_covariance
        else:
            mixing_warning = (
                "Too few non-negative, full-rank PSF draws; no PSF covariance contribution was added."
            )
    # Regularization stabilizes a numerical candidate; it must never create
    # measurement information in spatial null-space modes. The nominal data-only
    # identifiability decision above is independent of the curvature penalty.
    column_norms = np.linalg.norm(a, axis=0)
    scaled_data_operator = a / np.maximum(column_norms, np.finfo(float).tiny)[None, :]
    data_singular_values = np.linalg.svd(scaled_data_operator, compute_uv=False)
    data_tolerance = (
        np.finfo(float).eps * max(scaled_data_operator.shape)
        * (float(data_singular_values[0]) if data_singular_values.size else 0.0)
    )
    data_only_rank = int(np.count_nonzero(data_singular_values > data_tolerance))
    data_only_condition_number = (
        float(data_singular_values[0] / data_singular_values[-1])
        if data_singular_values.size and data_singular_values[-1] > data_tolerance
        else float("inf")
    )
    data_identifiable = (
        data_only_rank == x.size
        and np.isfinite(data_only_condition_number)
        and data_only_condition_number < 1.0e12
    )
    primary_status = (
        "identified_from_measurement"
        if data_identifiable else "conditional_regularized_candidate"
    )
    candidate_uncertainty = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    return {
        "analysis_status": "completed" if data_identifiable else "conditional_regularized_candidate",
        "primary_result_status": primary_status,
        "primary_output_reportable": bool(data_identifiable),
        "data_only_rank": data_only_rank,
        "latent_node_count": int(x.size),
        "data_only_condition_number": data_only_condition_number,
        "parameter_name":str(parameter_name),"parameter_unit":str(parameter_unit),
        "radius_m":x.copy(),
        "profile":solution.copy() if data_identifiable else None,
        "profile_uncertainty":candidate_uncertainty.copy() if data_identifiable else None,
        "profile_covariance":pcov.copy() if data_identifiable else None,
        "regularized_candidate_profile":solution.copy(),
        "regularized_candidate_uncertainty":candidate_uncertainty.copy(),
        "regularized_candidate_covariance":pcov.copy(),
        "measurement_conditional_profile_covariance":measurement_profile_covariance,
        "coordinate_induced_profile_covariance":coordinate_profile_covariance,
        "spatial_mixing_covariance":None if mixing_covariance is None else mixing_covariance.copy(),
        "spatial_mixing_induced_profile_covariance":mixing_profile_covariance,
        "spatial_mixing_uncertainty_propagation_status":(
            "not_supplied" if mixing_covariance is None else
            "propagated_by_monte_carlo_with_fixed_selected_regularization"
            if mixing_profile_covariance is not None else "insufficient_valid_draws"
        ),
        "spatial_mixing_uncertainty_successful_draws":int(mixing_successful_draws),
        "spatial_mixing_uncertainty_warning":mixing_warning,
        "radius_covariance_m2":None if coordinate_covariance is None else coordinate_covariance.copy(),
        "coordinate_uncertainty_propagation_status":(
            "not_supplied" if coordinate_covariance is None else
            "propagated_by_monte_carlo_with_fixed_selected_regularization"
            if coordinate_profile_covariance is not None else "insufficient_valid_draws"
        ),
        "coordinate_uncertainty_successful_draws":int(coordinate_successful_draws),
        "coordinate_uncertainty_warning":coordinate_warning,
        "measured_channel_values":y.copy(),"fitted_channel_values":h@solution,
        "residual_whitened":resid,"spatial_mixing_operator":"researcher_supplied_fixed_forward_operator",
        "spatial_mixing_matrix":h.copy(),
        "regularization":{"kind":"nonuniform_grid_second_derivative_curvature_penalty","strength":lam,
            "selection":selection,"gcv_score":gcv_score,"effective_degrees_of_freedom":dof},
        "covariance_model":covariance_model,"jacobian_rank":int(np.linalg.matrix_rank(a)),
        "jacobian_condition_number":float(np.linalg.cond(a)),"active_nonnegative_nodes":np.flatnonzero(active).tolist(),
        "uncertainty_semantics":("measurement covariance conditional on the active set, plus separately propagated coordinate and/or PSF covariance components when supplied"
            if positive else "linearized measurement covariance plus separately propagated coordinate and/or PSF covariance components when supplied"),
        "identifiability":{"primary":["regularized profile within supplied finite-dimensional spatial basis"],
            "conditional":["physical units/meaning are inherited from the upstream inferred parameter"],
            "non_identifiable":["spatial structures in the null space of the supplied mixing operator",
                "features finer than spatial resolution encoded by the supplied mixing operator"]},
        "experimental_validation":"not_performed; software/numerical inference only"
    }
