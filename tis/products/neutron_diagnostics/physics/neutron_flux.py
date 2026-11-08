"""Neutron detector count-rate and flux estimates with explicit time semantics."""
import numpy as np
from .counting import count_rate_from_timestamps


def analyze_neutron_flux(time_s, counts, *, timestamp_semantics=None, efficiency=None,
                         effective_area_m2=None, live_time_s=None,
                         counts_are_cumulative=False):
    """Estimate count rate and optionally flux from detector counts.

    timestamp_semantics must be "bin_centers", "bin_edges", or "counter_read_times".
    The meaning is never inferred from the numerical values of the counter.
    """
    counted = count_rate_from_timestamps(
        time_s, counts, timestamp_semantics=timestamp_semantics,
        counts_are_cumulative=counts_are_cumulative, live_time_s=live_time_s)
    out = dict(counted)
    out["counts"] = np.asarray(counts, dtype=float)
    if efficiency is not None or effective_area_m2 is not None:
        if efficiency is None or effective_area_m2 is None:
            raise ValueError("Provide detector efficiency and effective area")
        eff, area = float(efficiency), float(effective_area_m2)
        if not np.isfinite(eff) or not np.isfinite(area) or not 0 < eff <= 1 or area <= 0:
            raise ValueError("Efficiency must be finite in (0,1] and area finite positive")
        flux = out["count_rate_s-1"] / (eff * area)
        out["neutron_flux_m-2_s-1"] = flux
        out["flux_estimate_m-2_s-1"] = flux
        out["neutron_flux_m2_s"] = flux  # legacy API alias
    out["caveat"] = ("Count and timestamp semantics are explicit. Flux is an estimate and also requires "
                     "detector geometry, efficiency, background, dead-time and energy-response corrections.")
    return out


def neutron_flux_from_integrated_counts(counts, live_time_s, detector_efficiency, effective_area_m2):
    """Integrated-count flux estimator Phi=N/(T*epsilon*A) with Poisson counting uncertainty."""
    n, t, eff, area = np.broadcast_arrays(*[
        np.asarray(v, dtype=float)
        for v in (counts, live_time_s, detector_efficiency, effective_area_m2)])
    if any(not np.all(np.isfinite(v)) for v in (n, t, eff, area)) or np.any(n < 0) or np.any(t <= 0) or np.any(eff <= 0) or np.any(eff > 1) or np.any(area <= 0):
        raise ValueError("Counts non-negative; time/area positive; efficiency in (0,1]; inputs finite")
    denom = t * eff * area
    flux = n / denom
    uncertainty = np.sqrt(n) / denom
    return {"neutron_flux_m-2_s-1": flux, "flux_estimate_m-2_s-1": flux,
            "neutron_flux_m2_s": flux,
            "poisson_uncertainty_m-2_s-1": uncertainty, "poisson_uncertainty_m2_s": uncertainty,
            "equation": "Phi=N/(T*epsilon*A); sigma_Phi=sqrt(N)/(T*epsilon*A)",
            "assumption": "Poisson counting statistics and constant efficiency/area over live time; background and dead-time corrections must be applied separately."}
