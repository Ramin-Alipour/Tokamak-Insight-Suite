"""Reference-result comparison utilities.

Comparison decisions are only made against supplied uncertainty or tolerance
information. A bare numerical difference is reported, never silently labeled
as scientific agreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED = {"reference_value", "software_value"}


def compare_reference_results(
    table: pd.DataFrame,
    *,
    coverage_factor: float = 2.0,
) -> pd.DataFrame:
    """Compare computed outputs with independently supplied reference results.

    Required columns: reference_value, software_value.
    Optional columns: parameter, unit, source, reference_uncertainty,
    software_uncertainty, absolute_tolerance, relative_tolerance.

    Uncertainty agreement is assessed using |difference| <= k*sqrt(u_ref²+u_sw²).
    If both uncertainty columns are present, that rule takes precedence.
    Otherwise an explicit absolute tolerance, or relative tolerance multiplied
    by |reference|, may be used. Rows without a defensible supplied criterion
    are marked UNASSESSED rather than assigned an arbitrary pass/fail threshold.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame.")
    missing = _REQUIRED.difference(table.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if not np.isfinite(coverage_factor) or coverage_factor <= 0:
        raise ValueError("coverage_factor must be positive and finite.")

    out = table.copy()
    ref = pd.to_numeric(out["reference_value"], errors="coerce").to_numpy(float)
    got = pd.to_numeric(out["software_value"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        raise ValueError("Reference and software values must be finite numeric values.")
    delta = got - ref
    out["difference_software_minus_reference"] = delta
    out["absolute_difference"] = np.abs(delta)
    out["relative_difference"] = np.divide(
        delta, np.abs(ref), out=np.full(len(ref), np.nan), where=np.abs(ref) > 0
    )
    out["comparison_rule"] = "none"
    out["normalized_difference"] = np.nan
    out["agreement_status"] = "UNASSESSED"
    out["comparison_note"] = "No uncertainty or explicit tolerance supplied; numerical difference only."

    has_u_ref = "reference_uncertainty" in out.columns
    has_u_sw = "software_uncertainty" in out.columns
    if has_u_ref and has_u_sw:
        ur = pd.to_numeric(out["reference_uncertainty"], errors="coerce").to_numpy(float)
        us = pd.to_numeric(out["software_uncertainty"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(ur)) or not np.all(np.isfinite(us)) or np.any(ur < 0) or np.any(us < 0):
            raise ValueError("Uncertainties must be finite, numeric and nonnegative.")
        combined = np.hypot(ur, us)
        out["combined_standard_uncertainty"] = combined
        out["comparison_rule"] = f"combined_uncertainty_{coverage_factor:g}sigma"
        nonzero = combined > 0
        norm = np.divide(np.abs(delta), combined, out=np.full(len(ref), np.nan), where=nonzero)
        out["normalized_difference"] = norm
        agree = nonzero & (np.abs(delta) <= coverage_factor * combined)
        exact = (~nonzero) & (delta == 0)
        out["agreement_status"] = np.where(agree | exact, "AGREES_WITHIN_UNCERTAINTY", "DISAGREES_WITH_UNCERTAINTY")
        out["comparison_note"] = np.where(
            nonzero,
            f"Compared with {coverage_factor:g} times the combined standard uncertainty; this is not proof of model validity.",
            "Both supplied uncertainties are zero; only exact numerical equality is accepted."
        )
    else:
        abs_tol = pd.to_numeric(out["absolute_tolerance"], errors="coerce").to_numpy(float) if "absolute_tolerance" in out.columns else np.full(len(ref), np.nan)
        rel_tol = pd.to_numeric(out["relative_tolerance"], errors="coerce").to_numpy(float) if "relative_tolerance" in out.columns else np.full(len(ref), np.nan)
        if np.any(np.isfinite(abs_tol) & (abs_tol < 0)) or np.any(np.isfinite(rel_tol) & (rel_tol < 0)):
            raise ValueError("Tolerances must be nonnegative.")
        use_abs = np.isfinite(abs_tol)
        use_rel = (~use_abs) & np.isfinite(rel_tol)
        tol = np.where(use_abs, abs_tol, np.where(use_rel, rel_tol * np.abs(ref), np.nan))
        evaluable = np.isfinite(tol)
        out["comparison_rule"] = np.where(use_abs, "absolute_tolerance", np.where(use_rel, "relative_tolerance", "none"))
        out["normalized_difference"] = np.divide(np.abs(delta), tol, out=np.full(len(ref), np.nan), where=evaluable & (tol > 0))
        agree = evaluable & (np.abs(delta) <= tol)
        out["agreement_status"] = np.where(~evaluable, "UNASSESSED", np.where(agree, "AGREES_WITH_TOLERANCE", "DISAGREES_WITH_TOLERANCE"))
        out["comparison_note"] = np.where(evaluable, "Compared with the explicit user-supplied tolerance; this does not establish experimental validity.", "No uncertainty or explicit tolerance supplied; numerical difference only.")
        out["applied_tolerance"] = tol
    out["comparison_coverage_factor"] = float(coverage_factor)
    out["reference_value"] = ref
    out["software_value"] = got
    return out


def compare_reference_series(
    table: pd.DataFrame,
    *,
    coverage_factor: float = 2.0,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Compare digitized/reference curve data with a software-result series.

    Required columns are x_reference, reference_value, x_software,
    software_value. Optional uncertainty columns are sampled on their
    respective series grids. The reference curve is linearly interpolated
    onto software x values only over the common domain; no extrapolation is
    performed. Uncertainty interpolation assumes independent endpoint errors.
    """
    required = {"x_reference", "reference_value", "x_software", "software_value"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError("Missing required curve columns: " + ", ".join(sorted(missing)))
    if not np.isfinite(coverage_factor) or coverage_factor <= 0:
        raise ValueError("coverage_factor must be positive and finite.")
    if absolute_tolerance is not None and relative_tolerance is not None:
        raise ValueError("Specify absolute_tolerance or relative_tolerance, not both.")
    for name, value in (("absolute_tolerance", absolute_tolerance), ("relative_tolerance", relative_tolerance)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")

    xr = pd.to_numeric(table["x_reference"], errors="coerce").to_numpy(float)
    yr = pd.to_numeric(table["reference_value"], errors="coerce").to_numpy(float)
    xs = pd.to_numeric(table["x_software"], errors="coerce").to_numpy(float)
    ys = pd.to_numeric(table["software_value"], errors="coerce").to_numpy(float)
    mr = np.isfinite(xr) & np.isfinite(yr)
    ms = np.isfinite(xs) & np.isfinite(ys)
    xr, yr, xs, ys = xr[mr], yr[mr], xs[ms], ys[ms]
    if len(xr) < 2 or len(xs) < 2:
        raise ValueError("Each reference/software curve needs at least two finite coordinate-value pairs.")
    ir, isw = np.argsort(xr), np.argsort(xs)
    xr, yr, xs, ys = xr[ir], yr[isw], xs[isw], ys[isw]
    if np.any(np.diff(xr) <= 0) or np.any(np.diff(xs) <= 0):
        raise ValueError("Coordinates within each curve must be unique; duplicate x values must be resolved explicitly.")

    lo, hi = max(xr[0], xs[0]), min(xr[-1], xs[-1])
    overlap = (xs >= lo) & (xs <= hi)
    x = xs[overlap]
    software = ys[overlap]
    if len(x) < 2:
        raise ValueError("Reference and software curves overlap at fewer than two software coordinates.")
    reference = np.interp(x, xr, yr)
    difference = software - reference
    scale = float(np.max(reference) - np.min(reference))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    normalized_rmse = rmse / abs(scale) if abs(scale) > np.finfo(float).eps else None
    corr = float(np.corrcoef(reference, software)[0, 1]) if np.std(reference) > 0 and np.std(software) > 0 else None

    ref_unc = sw_unc = None
    if "reference_uncertainty" in table.columns and "software_uncertainty" in table.columns:
        ur_all = pd.to_numeric(table["reference_uncertainty"], errors="coerce").to_numpy(float)[mr][ir]
        us_all = pd.to_numeric(table["software_uncertainty"], errors="coerce").to_numpy(float)[ms][isw]
        if not np.all(np.isfinite(ur_all)) or not np.all(np.isfinite(us_all)) or np.any(ur_all < 0) or np.any(us_all < 0):
            raise ValueError("Curve uncertainties must be finite, nonnegative and supplied for every corresponding data point.")
        # Interpolate independent endpoint standard uncertainties in quadrature.
        right = np.searchsorted(xr, x, side="right").clip(1, len(xr)-1)
        left = right - 1
        weight = (x - xr[left]) / (xr[right] - xr[left])
        ref_unc = np.sqrt(((1.0-weight)*ur_all[left])**2 + (weight*ur_all[right])**2)
        sw_unc = us_all[overlap]
        combined = np.hypot(ref_unc, sw_unc)
        within = np.abs(difference) <= coverage_factor * combined
        rule = f"combined_uncertainty_{coverage_factor:g}sigma"
        normalized = np.divide(np.abs(difference), combined, out=np.full(len(x), np.nan), where=combined > 0)
        exact = (combined == 0) & (difference == 0)
        within = within | exact
    elif absolute_tolerance is not None:
        rule = "absolute_tolerance"
        normalized = np.divide(np.abs(difference), absolute_tolerance, out=np.full(len(x), np.nan), where=absolute_tolerance > 0)
        within = np.abs(difference) <= absolute_tolerance
    elif relative_tolerance is not None:
        rule = "relative_tolerance"
        tol = relative_tolerance * np.abs(reference)
        normalized = np.divide(np.abs(difference), tol, out=np.full(len(x), np.nan), where=tol > 0)
        within = np.abs(difference) <= tol
    else:
        rule = "none"
        normalized = np.full(len(x), np.nan)
        within = None

    if within is None:
        status = "UNASSESSED"
        fraction = None
    else:
        fraction = float(np.mean(within))
        status = "AGREES_WITHIN_CRITERION" if bool(np.all(within)) else "DISAGREES_WITH_CRITERION"

    aligned = pd.DataFrame({
        "x": x,
        "reference_value_interpolated": reference,
        "software_value": software,
        "difference_software_minus_reference": difference,
        "absolute_difference": np.abs(difference),
        "normalized_difference": normalized,
    })
    if ref_unc is not None:
        aligned["reference_uncertainty_interpolated"] = ref_unc
        aligned["software_uncertainty"] = sw_unc
    summary = {
        "n_reference_points": int(len(xr)),
        "n_software_points": int(len(xs)),
        "n_compared_points": int(len(x)),
        "x_overlap_min": float(lo),
        "x_overlap_max": float(hi),
        "bias_software_minus_reference": float(np.mean(difference)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": rmse,
        "normalized_RMSE_by_reference_range": normalized_rmse,
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "pearson_correlation": corr,
        "comparison_rule": rule,
        "fraction_within_criterion": fraction,
        "agreement_status": status,
        "coverage_factor": float(coverage_factor),
        "warning": "Reference interpolation is limited to the overlapping x-domain. Digitization, axis calibration, model and physical calibration uncertainties must be included by the analyst.",
    }
    return {"summary": summary, "aligned_comparison": aligned}
