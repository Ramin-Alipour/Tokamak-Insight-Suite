"""Finite-difference Grad-Shafranov equilibrium solver for prescribed profiles/boundary."""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


def _q95_from_profile(profile, q_profile):
    """Return q at normalized flux 0.95 only when supported by closed surfaces.

    np.interp silently clamps out-of-range requests to an endpoint. That would
    turn a last valid contour below 0.95 into a fabricated q95 estimate.
    """
    profile = np.asarray(profile, dtype=float)
    q = np.asarray(q_profile, dtype=float)
    if profile.ndim != 1 or q.shape != profile.shape:
        return None
    valid = np.isfinite(profile) & np.isfinite(q)
    x, y = profile[valid], q[valid]
    if x.size < 2 or not np.any(x <= 0.95) or not np.any(x >= 0.95):
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        return None
    return float(np.interp(0.95, x, y))


def _validated_grid(R_m, Z_m):
    R = np.asarray(R_m, dtype=float)
    Z = np.asarray(Z_m, dtype=float)
    if R.ndim != 1 or Z.ndim != 1 or R.size < 3 or Z.size < 3:
        raise ValueError("R and Z need at least 3 grid points")
    if (not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z))
            or np.any(R <= 0) or np.any(np.diff(R) <= 0)
            or np.any(np.diff(Z) <= 0)):
        raise ValueError("R must be positive and both grids strictly increasing and finite")
    if not np.allclose(np.diff(R), np.diff(R)[0]) or not np.allclose(np.diff(Z), np.diff(Z)[0]):
        raise ValueError("Uniform grids are required")
    return R, Z


def grad_shafranov_physical_relation(R_m, Z_m, psi_Wb_rad,
                                     pressure_gradient_Pa_per_Wb, ffprime_T2):
    """Evaluate Δ*ψ and residual Δ*ψ - [-μ0 R² p'(ψ) - F F'(ψ)]."""
    R = np.asarray(R_m, dtype=float)
    Z = np.asarray(Z_m, dtype=float)
    psi = np.asarray(psi_Wb_rad, dtype=float)
    dp = np.asarray(pressure_gradient_Pa_per_Wb, dtype=float)
    ff = np.asarray(ffprime_T2, dtype=float)
    if R.ndim != 1 or Z.ndim != 1 or R.size < 3 or Z.size < 3:
        raise ValueError("R and Z must be 1-D grids with at least three points")
    if (not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z))
            or np.any(R <= 0) or np.any(np.diff(R) <= 0) or np.any(np.diff(Z) <= 0)):
        raise ValueError("R must be positive and both grids strictly increasing and finite")
    expected = (Z.size, R.size)
    if any(a.shape != expected for a in (psi, dp, ff)):
        raise ValueError("psi, pressure-gradient, and FF-prime arrays must have shape (len(Z), len(R))")
    if not all(np.all(np.isfinite(a)) for a in (psi, dp, ff)):
        raise ValueError("All inputs must be finite")
    dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
    operator = (np.gradient(dpsi_dr, R, axis=1, edge_order=2)
                - dpsi_dr / R[None, :]
                + np.gradient(dpsi_dz, Z, axis=0, edge_order=2))
    source = -MU0 * R[None, :]**2 * dp - ff
    residual = operator - source
    return {
        "grad_shafranov_operator_Wb_rad_m-2": operator,
        "physical_source_Wb_rad_m-2": source,
        "equation_residual_Wb_rad_m-2": residual,
        "residual_rms_Wb_rad_m-2": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs_Wb_rad_m-2": float(np.max(np.abs(residual))),
        "equation": "Δ*ψ = -μ0 R² dp/dψ - F dF/dψ",
    }


def reconstruct_equilibrium(R_m, Z_m, boundary_flux_Wb_rad, psi_profile,
                            pressure_gradient_Pa_per_Wb, ffprime_T2, *,
                            max_iterations=10000, tolerance=1e-8, relaxation=1.2):
    """Solve Δ*ψ = -μ0 R²p'(ψ)-FF'(ψ) with fixed perimeter flux.

    The profiles are interpolated against min-max normalized flux of each iterate.
    This prescribed-boundary numerical solve is not a full experimental EFIT.
    """
    R, Z = _validated_grid(R_m, Z_m)
    bnd = np.asarray(boundary_flux_Wb_rad, dtype=float)
    prof = np.asarray(psi_profile, dtype=float)
    pp = np.asarray(pressure_gradient_Pa_per_Wb, dtype=float)
    ff = np.asarray(ffprime_T2, dtype=float)
    if bnd.shape != (Z.size, R.size) or not np.all(np.isfinite(bnd)):
        raise ValueError("Boundary flux must be finite with shape (len(Z), len(R))")
    if (prof.ndim != 1 or prof.size < 2 or pp.shape != prof.shape or ff.shape != prof.shape
            or not np.all(np.isfinite(prof)) or not np.all(np.isfinite(pp))
            or not np.all(np.isfinite(ff))):
        raise ValueError("Profile arrays must be finite 1-D arrays of equal length")
    if np.any(np.diff(prof) <= 0) or prof[0] < 0 or prof[-1] > 1:
        raise ValueError("psi_profile must increase strictly within [0,1]")
    if ((
            isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer, float, np.floating))
            or not np.isfinite(max_iterations)
            or int(max_iterations) != max_iterations
            or max_iterations < 1
        ) or (
            isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ) or (
            isinstance(relaxation, (bool, np.bool_))
            or not isinstance(relaxation, (int, float, np.integer, np.floating))
            or not np.isfinite(relaxation)
            or not 0 < relaxation < 2
        )):
        raise ValueError("Solver controls require a positive integer iteration limit, finite positive tolerance, and finite relaxation in (0,2)")
    psi = np.zeros_like(bnd)
    psi[0, :] = bnd[0, :]
    psi[-1, :] = bnd[-1, :]
    psi[:, 0] = bnd[:, 0]
    psi[:, -1] = bnd[:, -1]
    dr, dz = R[1] - R[0], Z[1] - Z[0]
    ar, az = 1.0 / dr**2, 1.0 / dz**2
    den = -2 * ar - 2 * az
    converged, change = False, np.inf
    for it in range(1, int(max_iterations) + 1):
        old = psi.copy()
        lo, hi = float(np.min(psi)), float(np.max(psi))
        span = hi - lo
        psin = np.zeros_like(psi) if span <= np.finfo(float).tiny else (psi - lo) / span
        source = -MU0 * R[None, :]**2 * np.interp(psin, prof, pp, left=pp[0], right=pp[-1]) - np.interp(psin, prof, ff, left=ff[0], right=ff[-1])
        for j in range(1, Z.size - 1):
            for i in range(1, R.size - 1):
                # Δ*ψ = ψ_RR - (1/R)ψ_R + ψ_ZZ; the first-derivative
                # term therefore reduces the right-neighbour coefficient.
                rp = ar - 1 / (2 * R[i] * dr)
                rm = ar + 1 / (2 * R[i] * dr)
                candidate = (source[j, i] - rp * psi[j, i+1] - rm * psi[j, i-1] - az * (psi[j+1, i] + psi[j-1, i])) / den
                psi[j, i] = (1 - relaxation) * psi[j, i] + relaxation * candidate
        change = float(np.max(np.abs(psi - old))) / max(float(np.max(np.abs(psi))), np.finfo(float).tiny)
        if change < tolerance:
            converged = True
            break
    dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
    br, bz = -dpsi_dz / R[None, :], dpsi_dr / R[None, :]
    span = float(np.max(psi) - np.min(psi))
    psin = np.zeros_like(psi) if span == 0 else (psi - np.min(psi)) / span
    # Re-evaluate source using the final normalized flux; report physical residual
    # only on interior cells, where the PDE is actually solved.
    final_source = -MU0 * R[None, :]**2 * np.interp(psin, prof, pp, left=pp[0], right=pp[-1]) - np.interp(psin, prof, ff, left=ff[0], right=ff[-1])
    physical = grad_shafranov_physical_relation(R, Z, psi, -final_source / (MU0 * R[None, :]**2), np.zeros_like(psi))
    # Use direct operator/source residual with the actual mapped source.
    operator = physical["grad_shafranov_operator_Wb_rad_m-2"]
    residual = operator - final_source
    interior = residual[1:-1, 1:-1]
    residual_rms = float(np.sqrt(np.mean(interior**2)))
    residual_scale = max(
        float(np.sqrt(np.mean(operator[1:-1, 1:-1] ** 2))),
        float(np.sqrt(np.mean(final_source[1:-1, 1:-1] ** 2))),
        float(np.max(np.abs(psi))) / min(dr, dz) ** 2,
        np.finfo(float).tiny,
    )
    relative_residual_rms = residual_rms / residual_scale
    # A small update alone can indicate stagnation, not a solution of the PDE.
    # Only advertise convergence when both the iteration and equation residual
    # satisfy their tolerances on the cells where the PDE is solved.
    converged = bool(converged and relative_residual_rms <= max(10.0 * tolerance, 1e-10))
    return {
        "R_m": R.copy(), "Z_m": Z.copy(), "psi_Wb_rad": psi,
        "psi_normalized": psin, "B_R_T": br, "B_Z_T": bz,
        "B_poloidal_T": np.hypot(br, bz), "iterations": it,
        "relative_iteration_change": change, "converged": converged,
        "equation_residual_rms_Wb_rad_m-2": residual_rms,
        "equation_residual_relative_rms": relative_residual_rms,
        "equation_residual_max_abs_Wb_rad_m-2": float(np.max(np.abs(interior))),
        "warning": "Prescribed-boundary/profile solve only; inspect PDE residual and validate against independent magnetic constraints. Not a full EFIT reconstruction.",
    }




def infer_grad_shafranov_profiles_from_flux(
    R_m, Z_m, reference_psi_Wb_rad, *, n_profile_points=32
):
    """Infer pressure-gradient and F F-prime source profiles from a flux map.

    This compatibility API now delegates to the joint R²-aware source fit.
    When the pressure and FF-prime terms are not identifiable from the map,
    the returned metadata explicitly marks the fallback source decomposition.
    """
    profiles = reconstruct_equilibrium_profiles_from_flux(
        R_m, Z_m, reference_psi_Wb_rad, n_profile_points=n_profile_points
    )
    return {
        "psi_profile": profiles["psi_normalized_axis_to_edge"],
        "pressure_gradient_Pa_per_Wb": profiles["pressure_gradient_Pa_per_Wb"],
        "ffprime_T2": profiles["ffprime_T2"],
        "reference_operator_Wb_rad_m-2": profiles["operator_Wb_rad_m-2"],
        "normalized_flux": profiles["psi_normalized_grid"],
        "profile_bin_counts": profiles["profile_bin_counts"],
        "source_model": profiles["profile_source_model"],
        "joint_pressure_ffprime_identifiable": profiles["joint_pressure_ffprime_identifiable"],
        "profile_bin_design_rank": profiles["profile_bin_design_rank"],
    }

def reconstruct_equilibrium_from_reference_flux(
    R_m, Z_m, reference_psi_Wb_rad, *, n_profile_points=32,
    max_iterations=10000, tolerance=1e-8, relaxation=1.2,
):
    """Run a data-backed Grad-Shafranov solve using a public/reference flux map.

    The reference supplies the fixed perimeter and a flux-binned source estimate.
    Returned error metrics compare the computed interior against that reference;
    because the source is estimated from the reference, this is a reconstruction
    workflow demonstration, not blind validation or an independent EFIT result.
    """
    R, Z = _validated_grid(R_m, Z_m)
    reference = np.asarray(reference_psi_Wb_rad, dtype=float)
    if reference.shape != (Z.size, R.size) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_psi_Wb_rad must be a finite (len(Z), len(R)) grid")
    profiles = reconstruct_equilibrium_profiles_from_flux(
        R, Z, reference, n_profile_points=n_profile_points
    )
    result = reconstruct_equilibrium(
        R, Z, reference, profiles["psi_normalized_axis_to_edge"],
        profiles["pressure_gradient_Pa_per_Wb"], profiles["ffprime_T2"],
        max_iterations=max_iterations, tolerance=tolerance, relaxation=relaxation,
    )
    difference = result["psi_Wb_rad"][1:-1, 1:-1] - reference[1:-1, 1:-1]
    ref_interior = reference[1:-1, 1:-1]
    scale = max(float(np.ptp(ref_interior)), np.finfo(float).tiny)
    result.update({
        "reference_comparison": {
            "interior_flux_rmse_Wb_rad": float(np.sqrt(np.mean(difference**2))),
            "interior_flux_nrmse_by_reference_range": float(np.sqrt(np.mean(difference**2)) / scale),
            "interior_flux_mae_Wb_rad": float(np.mean(np.abs(difference))),
            "reference_grid_shape": [int(Z.size), int(R.size)],
            "source_profile_model": profiles["profile_source_model"],
            "profile_bin_counts": profiles["profile_bin_counts"].tolist(),
            "joint_pressure_ffprime_identifiable": profiles["joint_pressure_ffprime_identifiable"],
            "profile_bin_design_rank": profiles["profile_bin_design_rank"].tolist(),
        },
        "inferred_profiles": {
            "psi_profile": profiles["psi_normalized_axis_to_edge"].tolist(),
            "pressure_gradient_Pa_per_Wb": profiles["pressure_gradient_Pa_per_Wb"].tolist(),
            "ffprime_T2": profiles["ffprime_T2"].tolist(),
            "pressure_Pa_relative_to_edge": profiles["pressure_Pa_relative_to_edge"].tolist(),
        },
        "reference_psi_Wb_rad": reference.copy(),
        "warning": (
            "Data-backed Grad-Shafranov reconstruction from a reference flux map. "
            "The joint p'(psi)/F F'(psi) source profile was estimated from that same map; "
            "comparison metrics are in-sample reconstruction errors, not independent validation. "
            "Profile decomposition is underdetermined when the per-flux-bin R² design is rank deficient."
        ),
    })
    return result

def reconstruct_equilibrium_inverse(
    R_m, Z_m, basis_psi_Wb_rad, *,
    magnetic_probe_response_T_per_coefficient=None,
    magnetic_probe_values_T=None,
    magnetic_probe_sigma_T=None,
    flux_loop_response_Wb_rad_per_coefficient=None,
    flux_loop_values_Wb_rad=None,
    flux_loop_sigma_Wb_rad=None,
    coil_probe_response_T_per_A=None,
    coil_probe_currents_A=None,
    coil_flux_response_Wb_rad_per_A=None,
    coil_flux_currents_A=None,
    regularization=1e-8,
):
    """Fit a linearized flux-basis equilibrium to *measured* magnetic data.

    The caller must provide calibrated sensor response matrices generated for the
    actual machine geometry. Basis coefficients are estimated by weighted,
    Tikhonov-regularized least squares. Known coil-current responses are removed
    from measurements before fitting. No synthetic measurements or machine
    geometry are generated here.

    This is an inverse magnetic fit framework, not a substitute for a machine-
    specific nonlinear EFIT implementation: the basis, response matrices, sensor
    calibration, and current constraints must be independently validated.
    """
    R, Z = _validated_grid(R_m, Z_m)
    basis = np.asarray(basis_psi_Wb_rad, dtype=float)
    if basis.ndim != 3 or basis.shape[1:] != (Z.size, R.size) or basis.shape[0] < 1:
        raise ValueError("basis_psi_Wb_rad must have shape (n_basis, len(Z), len(R))")
    if not np.all(np.isfinite(basis)):
        raise ValueError("Flux basis must be finite")
    if (
        isinstance(regularization, (bool, np.bool_))
        or not isinstance(regularization, (int, float, np.integer, np.floating))
        or not np.isfinite(regularization)
        or regularization < 0
    ):
        raise ValueError("regularization must be a finite non-negative number")
    n_basis = basis.shape[0]
    blocks, targets, weights = [], [], []

    def add_measurement_block(response, values, sigma, name, coil_response, coil_currents):
        if response is None and values is None and sigma is None:
            return
        if response is None or values is None or sigma is None:
            raise ValueError(f"{name}: response, measured values, and uncertainties are all required")
        matrix = np.asarray(response, dtype=float)
        observed = np.asarray(values, dtype=float)
        uncertainty = np.asarray(sigma, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != n_basis or observed.shape != (matrix.shape[0],) or uncertainty.shape != observed.shape:
            raise ValueError(f"{name}: incompatible response, value, or uncertainty dimensions")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(uncertainty)) or np.any(uncertainty <= 0):
            raise ValueError(f"{name}: all values must be finite and uncertainties positive")
        corrected = observed.copy()
        if coil_response is not None or coil_currents is not None:
            if coil_response is None or coil_currents is None:
                raise ValueError(f"{name}: coil response and coil currents must be supplied together")
            cr = np.asarray(coil_response, dtype=float)
            ci = np.asarray(coil_currents, dtype=float)
            if cr.shape != (matrix.shape[0], ci.size) or ci.ndim != 1 or not np.all(np.isfinite(cr)) or not np.all(np.isfinite(ci)):
                raise ValueError(f"{name}: invalid coil response/current dimensions or values")
            corrected -= cr @ ci
        blocks.append(matrix / uncertainty[:, None])
        targets.append(corrected / uncertainty)
        weights.append((name, matrix, observed, uncertainty, corrected))

    add_measurement_block(
        magnetic_probe_response_T_per_coefficient, magnetic_probe_values_T,
        magnetic_probe_sigma_T, "magnetic probes",
        coil_probe_response_T_per_A, coil_probe_currents_A)
    add_measurement_block(
        flux_loop_response_Wb_rad_per_coefficient, flux_loop_values_Wb_rad,
        flux_loop_sigma_Wb_rad, "flux loops",
        coil_flux_response_Wb_rad_per_A, coil_flux_currents_A)
    if not blocks:
        raise ValueError("At least one calibrated magnetic-probe or flux-loop measurement block is required")

    design = np.vstack(blocks)
    observed = np.concatenate(targets)
    # Scale columns to avoid unit/conditioning differences between basis functions.
    column_norm = np.linalg.norm(design, axis=0)
    if np.any(column_norm <= np.finfo(float).tiny):
        raise ValueError("At least one flux-basis coefficient is unobservable in the supplied measurements")
    scaled = design / column_norm
    # Regularization can make a singular system numerically solvable without making
    # its coefficients identifiable. Reject unobservable linear combinations before
    # regularization instead of returning an arbitrary-looking equilibrium.
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    rank_tolerance = (singular_values[0] * max(scaled.shape) * np.finfo(float).eps
                      if singular_values.size else 0.0)
    effective_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if effective_rank < n_basis:
        raise ValueError(
            f"Flux-basis coefficients are not identifiable from these measurements "
            f"(response rank {effective_rank} < {n_basis} coefficients); add independent "
            f"magnetic measurements or reduce the basis"
        )
    # Column scaling improves conditioning, but regularization is defined on the
    # physical basis coefficients. Since beta = coefficients * column_norm, the
    # Tikhonov rows must be diag(1 / column_norm), not the identity; otherwise
    # changing a basis function's units silently changes the regularization prior.
    penalty = np.diag(1.0 / column_norm)
    augmented = np.vstack((scaled, np.sqrt(regularization) * penalty))
    rhs = np.concatenate((observed, np.zeros(n_basis)))
    scaled_coefficients, *_ = np.linalg.lstsq(augmented, rhs, rcond=None)
    coefficients = scaled_coefficients / column_norm
    psi = np.einsum("k,kij->ij", coefficients, basis)
    if not np.all(np.isfinite(psi)) or float(np.ptp(psi)) <= np.finfo(float).tiny:
        raise ValueError("Inverse fit produced a degenerate flux field; check response matrices and data")

    # Determine the magnetic axis from the strongest interior extremum relative
    # to the computational-edge flux level, then infer a boundary flux level
    # from the edge. This is a data-derived estimate, not a machine limiter model.
    edge = np.concatenate((psi[0, :], psi[-1, :], psi[1:-1, 0], psi[1:-1, -1]))
    edge_flux = float(np.median(edge))
    interior = psi[1:-1, 1:-1]
    if abs(float(np.min(interior)) - edge_flux) >= abs(float(np.max(interior)) - edge_flux):
        local = np.unravel_index(np.argmin(interior), interior.shape)
        axis_flux = float(np.min(interior))
    else:
        local = np.unravel_index(np.argmax(interior), interior.shape)
        axis_flux = float(np.max(interior))
    axis_j, axis_i = int(local[0] + 1), int(local[1] + 1)
    denom = edge_flux - axis_flux
    if abs(denom) <= np.finfo(float).tiny:
        raise ValueError("Cannot infer a distinct plasma axis and edge flux from this reconstruction")
    psi_norm = (psi - axis_flux) / denom
    boundary_level = edge_flux
    boundary_mask = np.isclose(psi, boundary_level, rtol=0.02, atol=0.02 * max(abs(denom), np.finfo(float).tiny))
    # Extract actual level-set polylines from the fitted flux grid. This is a
    # computational-edge-flux contour estimate, not a machine-defined LCFS.
    from matplotlib.figure import Figure
    contour_figure = Figure()
    contour_axis = contour_figure.subplots()
    contour_set = contour_axis.contour(R, Z, psi, levels=[boundary_level])
    boundary_contours = [
        np.asarray(segment, dtype=float).tolist()
        for segment in contour_set.allsegs[0]
        if len(segment) >= 3
    ]
    contour_figure.clear()
    # Per-block residuals are reported in native units and uncertainty-normalized units.
    diagnostics = {}
    for name, matrix, measured, sigma, corrected in weights:
        prediction = matrix @ coefficients
        residual = measured - (prediction + (measured - corrected))
        diagnostics[name] = {
            "count": int(measured.size),
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
            "normalized_residual_rms": float(np.sqrt(np.mean((residual / sigma)**2))),
            "predicted_without_coils": prediction.tolist(),
            "measured": measured.tolist(),
        }
    return {
        "R_m": R.copy(), "Z_m": Z.copy(), "psi_Wb_rad": psi,
        "psi_normalized": psi_norm, "basis_coefficients": coefficients,
        "axis_R_m": float(R[axis_i]), "axis_Z_m": float(Z[axis_j]),
        "axis_flux_Wb_rad": axis_flux, "boundary_flux_Wb_rad": boundary_level,
        "boundary_mask": boundary_mask, "boundary_contours_RZ_m": boundary_contours,
        "edge_flux_estimator": "median of reconstructed computational-edge flux; not a machine-defined LCFS",
        "measurement_diagnostics": diagnostics,
        "weighted_residual_rms": float(np.linalg.norm(design @ coefficients - observed) / np.sqrt(observed.size)),
        "regularization": float(regularization),
        "measurement_count": int(observed.size),
        "warning": "Inverse magnetic fit using caller-supplied calibrated machine response. Validate basis, coil responses, inferred boundary, and Grad-Shafranov/profile consistency for the specific device; this generic linearized fit is not a complete machine-specific EFIT.",
    }



def derive_equilibrium_profiles_from_sources(
    R_m, Z_m, psi_Wb_rad, psi_profile,
    pressure_gradient_Pa_per_Wb, ffprime_T2, *,
    toroidal_field_boundary_Tm=None,
):
    """Derive pressure/current profiles from the *supplied* GS source profiles.

    Unlike reconstruct_equilibrium_profiles_from_flux, this routine does not
    refit p' and FF' from the solved flux map. It preserves the user's supplied
    source functions, integrates p(edge)=0, and computes j_phi. If an independent
    edge F=R B_phi is supplied, it also integrates F^2. This is a profile
    derivation for a prescribed-profile GS solution, not free-boundary EFIT.
    """
    R, Z = _validated_grid(R_m, Z_m)
    psi = np.asarray(psi_Wb_rad, dtype=float)
    profile = np.asarray(psi_profile, dtype=float)
    pprime_input = np.asarray(pressure_gradient_Pa_per_Wb, dtype=float)
    ffprime_input = np.asarray(ffprime_T2, dtype=float)
    if psi.shape != (Z.size, R.size) or not np.all(np.isfinite(psi)):
        raise ValueError("psi_Wb_rad must be a finite (len(Z), len(R)) grid")
    if (profile.ndim != 1 or profile.size < 2 or pprime_input.shape != profile.shape
            or ffprime_input.shape != profile.shape or not np.all(np.isfinite(profile))
            or not np.all(np.isfinite(pprime_input)) or not np.all(np.isfinite(ffprime_input))
            or np.any(np.diff(profile) <= 0) or profile[0] < 0 or profile[-1] > 1):
        raise ValueError("Source profiles must be finite, aligned, and strictly increasing within [0, 1]")
    if toroidal_field_boundary_Tm is not None and (
        not np.isscalar(toroidal_field_boundary_Tm)
        or not np.isfinite(toroidal_field_boundary_Tm)
        or float(toroidal_field_boundary_Tm) == 0
    ):
        raise ValueError("toroidal_field_boundary_Tm must be finite and non-zero when supplied")
    edge = np.concatenate((psi[0, :], psi[-1, :], psi[1:-1, 0], psi[1:-1, -1]))
    edge_flux = float(np.median(edge))
    interior = psi[1:-1, 1:-1]
    if abs(float(np.max(psi) - np.min(psi))) <= np.finfo(float).tiny:
        raise ValueError("Solved flux map has no finite flux span")
    axis_offset_min = abs(float(np.min(interior)) - edge_flux)
    axis_offset_max = abs(float(np.max(interior)) - edge_flux)
    axis_flux = float(np.min(interior) if axis_offset_min >= axis_offset_max else np.max(interior))
    axis_index = np.unravel_index(np.argmin(interior) if axis_offset_min >= axis_offset_max else np.argmax(interior), interior.shape)
    axis_j, axis_i = int(axis_index[0] + 1), int(axis_index[1] + 1)
    flux_span = edge_flux - axis_flux
    if abs(flux_span) <= np.finfo(float).tiny:
        raise ValueError("Solved flux has no distinct axis-to-edge flux span")
    psi_n_grid = (psi - axis_flux) / flux_span
    profile = np.asarray(profile, dtype=float)
    pprime = np.interp(profile, np.asarray(psi_profile, dtype=float), pprime_input)
    ffprime = np.interp(profile, np.asarray(psi_profile, dtype=float), ffprime_input)
    pressure = np.zeros_like(profile)
    for k in range(profile.size - 2, -1, -1):
        pressure[k] = pressure[k + 1] - 0.5 * (
            pprime[k + 1] + pprime[k]
        ) * flux_span * (profile[k + 1] - profile[k])
    rr = np.broadcast_to(R[None, :], psi.shape)
    p_grid = np.interp(np.clip(psi_n_grid, 0.0, 1.0), profile, pprime)
    ff_grid = np.interp(np.clip(psi_n_grid, 0.0, 1.0), profile, ffprime)
    j_tor = rr * p_grid + ff_grid / (MU0 * rr)
    j_tor[(psi_n_grid < 0.0) | (psi_n_grid > 1.0)] = np.nan
    result = {
        "psi_normalized_axis_to_edge": profile,
        "pressure_gradient_Pa_per_Wb": pprime,
        "ffprime_T2": ffprime,
        "pressure_Pa_relative_to_edge": pressure,
        "j_tor_A_m2": j_tor,
        "psi_axis_Wb_rad": axis_flux,
        "psi_edge_Wb_rad": edge_flux,
        "axis_R_m": float(R[axis_i]),
        "axis_Z_m": float(Z[axis_j]),
        "psi_normalized_grid": psi_n_grid,
        "profile_source_model": "directly supplied p'(psi_N) and F F'(psi_N); no inverse source refit",
        "pressure_profile_is_relative": True,
    }
    if toroidal_field_boundary_Tm is not None:
        f2 = np.empty_like(profile)
        f2[-1] = float(toroidal_field_boundary_Tm) ** 2
        integrand = 2.0 * ffprime * flux_span
        for k in range(profile.size - 2, -1, -1):
            f2[k] = f2[k + 1] - 0.5 * (integrand[k + 1] + integrand[k]) * (profile[k + 1] - profile[k])
        if np.any(f2 <= 0) or not np.all(np.isfinite(f2)):
            raise ValueError("Inferred F²(psi) is non-positive; check flux convention and edge F")
        result["F_squared_T2_m2"] = f2
        F = np.sign(float(toroidal_field_boundary_Tm)) * np.sqrt(f2)
        result["F_Tm"] = F
        # Safety factor from closed flux contours, only where a contour around
        # the inferred axis exists and the supplied edge F makes F(psi) defined.
        from matplotlib.figure import Figure
        from matplotlib.path import Path as PolygonPath
        from scipy.interpolate import RegularGridInterpolator

        dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
        grad_mag = np.hypot(dpsi_dr, dpsi_dz)
        grad_interp = RegularGridInterpolator((Z, R), grad_mag, bounds_error=False, fill_value=np.nan)
        fig = Figure()
        ax = fig.subplots()
        q_profile = np.full(profile.size, np.nan)
        for k, level_n in enumerate(profile):
            if level_n <= 0.02 or level_n >= 0.995:
                continue
            level = axis_flux + level_n * flux_span
            contours = ax.contour(R, Z, psi, levels=[level])
            segments = contours.allsegs[0]
            chosen = None
            for segment in segments:
                if len(segment) < 8 or not np.allclose(segment[0], segment[-1], rtol=0, atol=1e-6):
                    continue
                if PolygonPath(segment).contains_point((R[axis_i], Z[axis_j])):
                    if chosen is None or len(segment) > len(chosen):
                        chosen = segment
            for collection in getattr(contours, "collections", []):
                collection.remove()
            if chosen is None:
                continue
            points = np.asarray(chosen, dtype=float)
            delta = np.diff(points, axis=0)
            dl = np.hypot(delta[:, 0], delta[:, 1])
            mid = 0.5 * (points[:-1] + points[1:])
            grad_mid = grad_interp(mid[:, ::-1])
            denominator = mid[:, 0] ** 2 * grad_mid
            good = np.isfinite(denominator) & (denominator > np.finfo(float).tiny)
            if np.count_nonzero(good) < 6:
                continue
            q_profile[k] = F[k] * float(np.sum(dl[good] / denominator[good])) / (2.0 * np.pi)
        fig.clear()
        result["q_profile"] = q_profile
        result["q95"] = _q95_from_profile(profile, q_profile)
        result["q_definition"] = "q(psi)=F(psi)/(2π) ∮ dl/[R² |∇psi|]; only closed contours containing the inferred magnetic axis are used"
    pressure_grid = np.interp(np.clip(psi_n_grid, 0.0, 1.0), profile, pressure)
    pressure_grid[(psi_n_grid < 0.0) | (psi_n_grid > 1.0)] = np.nan
    result.update(derive_equilibrium_geometry_from_flux(
        R, Z, psi, psi_normalized_grid=psi_n_grid,
        pressure_grid=pressure_grid, j_tor_A_m2=j_tor,
    ))
    return result


def reconstruct_equilibrium_profiles_from_flux(
    R_m, Z_m, psi_Wb_rad, *, n_profile_points=32,
    toroidal_field_boundary_Tm=None,
):
    """Infer flux-surface p' and FF' source profiles from a 2-D flux map.

    The Grad–Shafranov source is decomposed in each normalized-flux bin as
    Δ*ψ = -μ0 R² p'(ψ) - F(ψ)F'(ψ). This is a constrained source fit, not a
    unique pressure/current reconstruction: the two functions are not
    identifiable from a single flux map unless the R² variation within a
    surface bin provides enough independent leverage. If the joint fit is
    poorly identified, this function reports that limitation rather than
    silently presenting p'=0 as measured physics.

    Pressure is integrated with p(edge)=0. F(ψ) and q(ψ) are calculated only
    when the user supplies F(edge)=R B_t(edge) in T·m. The q integral uses
    closed flux contours and the physical-flux gradient.
    """
    R, Z = _validated_grid(R_m, Z_m)
    psi = np.asarray(psi_Wb_rad, dtype=float)
    if psi.shape != (Z.size, R.size) or not np.all(np.isfinite(psi)):
        raise ValueError("psi_Wb_rad must be a finite (len(Z), len(R)) grid")
    if (isinstance(n_profile_points, (bool, np.bool_))
            or not isinstance(n_profile_points, (int, np.integer))
            or n_profile_points < 4 or n_profile_points > 512):
        raise ValueError("n_profile_points must be an integer from 4 to 512")
    if toroidal_field_boundary_Tm is not None and (
        not np.isscalar(toroidal_field_boundary_Tm)
        or not np.isfinite(toroidal_field_boundary_Tm)
        or float(toroidal_field_boundary_Tm) == 0
    ):
        raise ValueError("toroidal_field_boundary_Tm must be finite and non-zero when supplied")

    # The magnetic axis is the strongest interior extremum relative to the
    # median computational-edge flux; normalize axis-to-edge regardless of
    # whether the device uses increasing or decreasing psi.
    edge = np.concatenate((psi[0, :], psi[-1, :], psi[1:-1, 0], psi[1:-1, -1]))
    edge_flux = float(np.median(edge))
    interior = psi[1:-1, 1:-1]
    min_offset = abs(float(np.min(interior)) - edge_flux)
    max_offset = abs(float(np.max(interior)) - edge_flux)
    if min_offset >= max_offset:
        local = np.unravel_index(np.argmin(interior), interior.shape)
        axis_flux = float(np.min(interior))
    else:
        local = np.unravel_index(np.argmax(interior), interior.shape)
        axis_flux = float(np.max(interior))
    axis_j, axis_i = int(local[0] + 1), int(local[1] + 1)
    # psi_N = 0 at the magnetic axis and 1 at the computational edge,
    # irrespective of whether the machine's psi convention increases inward.
    flux_span = edge_flux - axis_flux
    if abs(flux_span) <= np.finfo(float).tiny:
        raise ValueError("Flux map must have a nonzero finite range and distinct interior magnetic-axis and edge flux")
    psi_n = (psi - axis_flux) / flux_span

    dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
    operator = (
        np.gradient(dpsi_dr, R, axis=1, edge_order=2)
        - dpsi_dr / R[None, :]
        + np.gradient(dpsi_dz, Z, axis=0, edge_order=2)
    )
    profile = np.linspace(0.0, 1.0, int(n_profile_points))
    edges = np.linspace(0.0, 1.0, int(n_profile_points) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pprime_bins = np.full(profile.size, np.nan)
    ffprime_bins = np.full(profile.size, np.nan)
    counts = np.zeros(profile.size, dtype=int)
    ranks = np.zeros(profile.size, dtype=int)
    rr = np.broadcast_to(R[None, :], psi.shape)
    valid = (psi_n[1:-1, 1:-1] >= 0.0) & (psi_n[1:-1, 1:-1] <= 1.0)
    svals = psi_n[1:-1, 1:-1][valid]
    yvals = operator[1:-1, 1:-1][valid]
    rvals = rr[1:-1, 1:-1][valid]
    bin_ids = np.clip(np.digitize(svals, edges[1:-1]), 0, profile.size - 1)
    for k in range(profile.size):
        mask = bin_ids == k
        counts[k] = int(np.count_nonzero(mask))
        if counts[k] < 3:
            continue
        # Fit the actual Grad–Shafranov source, including its R² dependence.
        design = np.column_stack((-MU0 * rvals[mask]**2, -np.ones(counts[k])))
        ranks[k] = int(np.linalg.matrix_rank(design))
        if ranks[k] == 2:
            fitted, *_ = np.linalg.lstsq(design, yvals[mask], rcond=None)
            pprime_bins[k], ffprime_bins[k] = fitted
    populated = np.flatnonzero(np.isfinite(pprime_bins) & np.isfinite(ffprime_bins))
    identifiable = populated.size >= 2
    if identifiable:
        pprime = np.interp(np.arange(profile.size), populated, pprime_bins[populated])
        ffprime = np.interp(np.arange(profile.size), populated, ffprime_bins[populated])
        model = "joint per-flux-bin fit of p'(psi) and F F'(psi) using R² variation"
    else:
        # Preserve a usable source estimate for tiny/degenerate grids, while
        # explicitly marking that pressure and FF' could not be separated.
        counts_by_bin = np.bincount(bin_ids, weights=yvals, minlength=profile.size)
        sums_by_bin = np.bincount(bin_ids, weights=np.ones_like(yvals), minlength=profile.size)
        source_mean = np.divide(counts_by_bin, sums_by_bin, out=np.full(profile.size, np.nan), where=sums_by_bin > 0)
        good = np.flatnonzero(np.isfinite(source_mean))
        if good.size < 2:
            raise ValueError("Not enough interior flux-surface samples to estimate source profiles")
        source_mean = np.interp(np.arange(profile.size), good, source_mean[good])
        pprime = np.zeros_like(profile)
        ffprime = -source_mean
        model = "fallback FF'-only source estimate; joint p'/FF' fit not identifiable on this grid"

    # psi = psi_edge + psi_N * (psi_axis - psi_edge); integrate from edge
    # inward so p(edge)=0 and F(edge) is an explicit user-provided constraint.
    dpsi_dpsi_n = flux_span
    p_integrand = pprime * dpsi_dpsi_n
    pressure = np.zeros_like(profile)
    for k in range(profile.size - 2, -1, -1):
        pressure[k] = pressure[k + 1] - 0.5 * (p_integrand[k + 1] + p_integrand[k]) * (profile[k + 1] - profile[k])
    # Reconstruct the local toroidal-current-density map from the fitted
    # Grad–Shafranov source functions: j_phi = R p'(psi) + F F'(psi)/(mu0 R).
    # Values outside the inferred axis-to-edge flux interval are deliberately NaN.
    psi_n_clipped = np.clip(psi_n, 0.0, 1.0)
    pprime_grid = np.interp(psi_n_clipped, profile, pprime)
    ffprime_grid = np.interp(psi_n_clipped, profile, ffprime)
    j_tor = rr * pprime_grid + ffprime_grid / (MU0 * rr)
    j_tor[~((psi_n >= 0.0) & (psi_n <= 1.0))] = np.nan
    result = {
        "psi_normalized_axis_to_edge": profile,
        "pressure_gradient_Pa_per_Wb": pprime,
        "ffprime_T2": ffprime,
        "j_tor_A_m2": j_tor,
        "pressure_Pa_relative_to_edge": pressure,
        "psi_axis_Wb_rad": axis_flux,
        "psi_edge_Wb_rad": edge_flux,
        "axis_R_m": float(R[axis_i]),
        "axis_Z_m": float(Z[axis_j]),
        "profile_bin_counts": counts,
        "profile_bin_design_rank": ranks,
        "joint_pressure_ffprime_identifiable": bool(identifiable),
        "profile_source_model": model,
        "operator_Wb_rad_m-2": operator,
        "psi_normalized_grid": psi_n,
        "pressure_nonnegative_everywhere": bool(np.all(pressure >= -1e-10 * max(float(np.max(np.abs(pressure))), 1.0))),
    }

    if toroidal_field_boundary_Tm is not None:
        f2 = np.empty_like(profile)
        f2[-1] = float(toroidal_field_boundary_Tm) ** 2
        integrand = 2.0 * ffprime * dpsi_dpsi_n
        for k in range(profile.size - 2, -1, -1):
            f2[k] = f2[k + 1] - 0.5 * (integrand[k + 1] + integrand[k]) * (profile[k + 1] - profile[k])
        if np.any(f2 <= 0) or not np.all(np.isfinite(f2)):
            raise ValueError("Inferred F²(psi) is non-positive; check flux convention, source fit, and supplied edge toroidal field")
        F = np.sign(float(toroidal_field_boundary_Tm)) * np.sqrt(f2)
        result["F_Tm"] = F
        result["F_squared_T2_m2"] = f2

        # q(psi) = F/(2π) ∮ dl/(R² |∇psi|) on each closed flux surface.
        from matplotlib.figure import Figure
        from matplotlib.path import Path as PolygonPath
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import map_coordinates

        grad_mag = np.hypot(dpsi_dr, dpsi_dz)
        grad_interp = RegularGridInterpolator((Z, R), grad_mag, bounds_error=False, fill_value=np.nan)
        fig = Figure()
        ax = fig.subplots()
        q_profile = np.full(profile.size, np.nan)
        for k, level_n in enumerate(profile):
            if level_n <= 0.02 or level_n >= 0.995:
                continue
            level = axis_flux + level_n * flux_span
            contours = ax.contour(R, Z, psi, levels=[level])
            segments = contours.allsegs[0]
            chosen = None
            for segment in segments:
                if len(segment) < 8 or not np.allclose(segment[0], segment[-1], rtol=0, atol=1e-6):
                    continue
                if PolygonPath(segment).contains_point((R[axis_i], Z[axis_j])):
                    if chosen is None or len(segment) > len(chosen):
                        chosen = segment
            for collection in getattr(contours, "collections", []):
                collection.remove()
            if chosen is None:
                continue
            points = np.asarray(chosen, dtype=float)
            delta = np.diff(points, axis=0)
            dl = np.hypot(delta[:, 0], delta[:, 1])
            mid = 0.5 * (points[:-1] + points[1:])
            grad_mid = grad_interp(mid[:, ::-1])
            denominator = mid[:, 0] ** 2 * grad_mid
            good = np.isfinite(denominator) & (denominator > np.finfo(float).tiny)
            if np.count_nonzero(good) < 6:
                continue
            integral = float(np.sum(dl[good] / denominator[good]))
            q_profile[k] = F[k] * integral / (2.0 * np.pi)
        fig.clear()
        result["q_profile"] = q_profile
        result["q95"] = _q95_from_profile(profile, q_profile)
        result["q_definition"] = "q(psi)=F(psi)/(2π) ∮ dl/[R² |∇psi|]; only closed contours containing the inferred magnetic axis are used"
    pressure_grid = np.interp(np.clip(psi_n, 0.0, 1.0), profile, pressure)
    pressure_grid[(psi_n < 0.0) | (psi_n > 1.0)] = np.nan
    result.update(derive_equilibrium_geometry_from_flux(
        R, Z, psi, psi_normalized_grid=psi_n,
        pressure_grid=pressure_grid, j_tor_A_m2=j_tor,
    ))
    return result


def derive_equilibrium_geometry_from_flux(
    R_m, Z_m, psi_Wb_rad, *, psi_normalized_grid=None,
    pressure_grid=None, j_tor_A_m2=None, shape_surface_psi_n=0.95,
):
    """Derive axis, closed-surface shaping, X-point and global estimates.

    Shaping metrics are evaluated on the requested normalized-flux contour
    (default 95%), not mislabeled as a measured separatrix. An X-point and
    separatrix flux are reported only when a near-edge saddle is detected.
    beta_p and li use axisymmetric 2πR volume weighting and the supplied
    pressure/current maps; they remain estimates tied to those inputs.
    """
    R, Z = _validated_grid(R_m, Z_m)
    psi = np.asarray(psi_Wb_rad, dtype=float)
    if psi.shape != (Z.size, R.size) or not np.all(np.isfinite(psi)):
        raise ValueError("psi_Wb_rad must be a finite (len(Z), len(R)) grid")
    if not np.isscalar(shape_surface_psi_n) or not np.isfinite(shape_surface_psi_n) or not 0.5 <= shape_surface_psi_n < 1.0:
        raise ValueError("shape_surface_psi_n must be finite and in [0.5, 1)")

    edge = np.concatenate((psi[0, :], psi[-1, :], psi[1:-1, 0], psi[1:-1, -1]))
    edge_flux = float(np.median(edge))
    interior = psi[1:-1, 1:-1]
    min_offset = abs(float(np.min(interior)) - edge_flux)
    max_offset = abs(float(np.max(interior)) - edge_flux)
    if min_offset >= max_offset:
        axis_flux = float(np.min(interior))
        axis_local = np.unravel_index(np.argmin(interior), interior.shape)
    else:
        axis_flux = float(np.max(interior))
        axis_local = np.unravel_index(np.argmax(interior), interior.shape)
    axis_j, axis_i = int(axis_local[0] + 1), int(axis_local[1] + 1)
    flux_span = edge_flux - axis_flux
    if abs(flux_span) <= np.finfo(float).tiny:
        raise ValueError("Flux map must have a nonzero finite axis-to-edge span")
    if psi_normalized_grid is None:
        psi_n = (psi - axis_flux) / flux_span
    else:
        psi_n = np.asarray(psi_normalized_grid, dtype=float)
        if psi_n.shape != psi.shape or not np.all(np.isfinite(psi_n)):
            raise ValueError("psi_normalized_grid must be finite and match psi_Wb_rad")

    result = {
        "axis_R_m": float(R[axis_i]),
        "axis_Z_m": float(Z[axis_j]),
        "psi_axis_Wb_rad": axis_flux,
        "psi_edge_Wb_rad": edge_flux,
        "shape_surface_psi_n": float(shape_surface_psi_n),
        "shape_surface_found": False,
        "major_radius_95_m": None,
        "minor_radius_95_m": None,
        "elongation_95": None,
        "triangularity_upper_95": None,
        "triangularity_lower_95": None,
        "xpoint_found": False,
        "xpoint_R_m": None,
        "xpoint_Z_m": None,
        "separatrix_found": False,
        "separatrix_flux_Wb_rad": None,
        "plasma_current_A": None,
        "beta_p": None,
        "li_internal": None,
        "geometry_interpretation": (
            "Shaping is measured on a closed 95%-flux contour. X-point/separatrix "
            "are reported only when a near-edge saddle is resolved. beta_p and li "
            "assume axisymmetry and the supplied pressure/current maps."
        ),
    }

    # Geometric shaping on a closed interior surface around the inferred axis.
    from matplotlib.figure import Figure
    from matplotlib.path import Path as PolygonPath
    from scipy.ndimage import minimum_filter

    fig = Figure()
    ax = fig.subplots()
    shape_level = axis_flux + float(shape_surface_psi_n) * flux_span
    contour_set = ax.contour(R, Z, psi, levels=[shape_level])
    chosen = None
    for segment in contour_set.allsegs[0]:
        if len(segment) < 8 or not np.allclose(segment[0], segment[-1], rtol=0, atol=1e-6):
            continue
        if PolygonPath(segment).contains_point((R[axis_i], Z[axis_j])):
            if chosen is None or len(segment) > len(chosen):
                chosen = segment
    for collection in getattr(contour_set, "collections", []):
        collection.remove()
    fig.clear()
    if chosen is not None:
        points = np.asarray(chosen, dtype=float)
        r_min, r_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
        z_min, z_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
        a_minor = 0.5 * (r_max - r_min)
        r_major = 0.5 * (r_max + r_min)
        if a_minor > 0:
            upper = points[int(np.argmax(points[:, 1]))]
            lower = points[int(np.argmin(points[:, 1]))]
            result.update({
                "shape_surface_found": True,
                "major_radius_95_m": r_major,
                "minor_radius_95_m": a_minor,
                "elongation_95": (z_max - z_min) / (2.0 * a_minor),
                "triangularity_upper_95": (r_major - float(upper[0])) / a_minor,
                "triangularity_lower_95": (r_major - float(lower[0])) / a_minor,
            })

    # A saddle near the edge is an X-point candidate. If the grid cannot
    # resolve one, do not manufacture a separatrix location.
    dpsi_dz, dpsi_dr = np.gradient(psi, Z, R, edge_order=2)
    d2r = np.gradient(dpsi_dr, R, axis=1, edge_order=2)
    d2z = np.gradient(dpsi_dz, Z, axis=0, edge_order=2)
    d2rz = np.gradient(dpsi_dr, Z, axis=0, edge_order=2)
    hessian_det = d2r * d2z - d2rz**2
    grad_mag = np.hypot(dpsi_dr, dpsi_dz)
    local_min = grad_mag <= minimum_filter(grad_mag, size=3, mode="nearest") + np.finfo(float).eps
    candidate_mask = np.zeros_like(psi, dtype=bool)
    candidate_mask[2:-2, 2:-2] = True
    candidate_mask &= (hessian_det < 0.0) & local_min & (psi_n >= 0.75) & (psi_n <= 1.25)
    candidate_mask[axis_j-2:axis_j+3, axis_i-2:axis_i+3] = False
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size:
        jx, ix = min(candidate_indices, key=lambda ij: grad_mag[int(ij[0]), int(ij[1])])
        result["xpoint_found"] = True
        result["xpoint_R_m"] = float(R[ix])
        result["xpoint_Z_m"] = float(Z[jx])
        x_flux = float(psi[jx, ix])
        if abs(float(psi_n[jx, ix]) - 1.0) <= 0.15:
            result["separatrix_found"] = True
            result["separatrix_flux_Wb_rad"] = x_flux

    # Global equilibrium estimates from axisymmetric volume integrals.
    if pressure_grid is not None or j_tor_A_m2 is not None:
        inside = (psi_n >= 0.0) & (psi_n <= 1.0)
        volume_weight = np.broadcast_to((2.0 * np.pi * R)[None, :], psi.shape)
        bp2 = (dpsi_dz / R[None, :])**2 + (dpsi_dr / R[None, :])**2

        def volume_integral(values):
            arr = np.where(inside & np.isfinite(values), values * volume_weight, 0.0)
            return float(np.trapezoid(np.trapezoid(arr, R, axis=1), Z))

        bp_energy_integral = volume_integral(bp2)
        if pressure_grid is not None:
            pgrid = np.asarray(pressure_grid, dtype=float)
            if pgrid.shape != psi.shape:
                raise ValueError("pressure_grid must match psi_Wb_rad")
            valid_p = inside & np.isfinite(pgrid)
            pscale = max(float(np.max(np.abs(pgrid[valid_p]))) if np.any(valid_p) else 0.0, 1.0)
            if np.any(valid_p) and np.all(pgrid[valid_p] >= -1e-10 * pscale) and bp_energy_integral > 0:
                p_volume_integral = volume_integral(pgrid)
                result["beta_p"] = max(0.0, 2.0 * MU0 * p_volume_integral / bp_energy_integral)

        if j_tor_A_m2 is not None:
            jgrid = np.asarray(j_tor_A_m2, dtype=float)
            if jgrid.shape != psi.shape:
                raise ValueError("j_tor_A_m2 must match psi_Wb_rad")
            jcross = np.where(inside & np.isfinite(jgrid), jgrid, 0.0)
            current = float(np.trapezoid(np.trapezoid(jcross, R, axis=1), Z))
            result["plasma_current_A"] = current
            if abs(current) > np.finfo(float).tiny and R[axis_i] > 0 and bp_energy_integral > 0:
                poloidal_energy = bp_energy_integral / (2.0 * MU0)
                result["li_internal"] = 4.0 * poloidal_energy / (MU0 * R[axis_i] * current**2)
    return result
