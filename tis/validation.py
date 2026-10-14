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
