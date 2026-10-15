"""Gas fueling and effective pumping-speed calculations with explicit units."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.integrate import cumulative_trapezoid

R_GAS = 8.31446261815324
N_AVOGADRO = 6.02214076e23


def gas_flow_to_particle_inventory(
    time_s: Any,
    flow_sccm: Any,
    *,
    reference_pressure_Pa: float = 101325.0,
    reference_temperature_K: float = 273.15,
) -> dict[str, Any]:
    """Convert standard cm³/min to molar/particle flow and integrated inventory.

    The reference pressure and temperature define what the flowmeter means by
    a standard cubic centimetre. The flow must already be calibrated and
    corrected for gas species and meter operating conditions.
    """
    t = np.asarray(time_s, dtype=float).ravel()
    q = np.asarray(flow_sccm, dtype=float).ravel()
    if len(t) != len(q) or len(t) < 2:
        raise ValueError("Time and flow arrays must have equal lengths and at least two samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
        raise ValueError("Time and flow values must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing without duplicate timestamps.")
    if np.any(q < 0):
        raise ValueError("Flow rates must be nonnegative; represent reverse flow separately.")
    if not np.isfinite(reference_pressure_Pa) or reference_pressure_Pa <= 0:
        raise ValueError("Reference pressure must be positive and finite.")
    if not np.isfinite(reference_temperature_K) or reference_temperature_K <= 0:
        raise ValueError("Reference temperature must be positive and finite.")
    q_m3_s = q * 1e-6 / 60.0
    molar_flow = reference_pressure_Pa * q_m3_s / (R_GAS * reference_temperature_K)
    particle_flow = molar_flow * N_AVOGADRO
    particles = cumulative_trapezoid(particle_flow, t, initial=0.0)
    moles = cumulative_trapezoid(molar_flow, t, initial=0.0)
    return {
        "time_s": t,
        "flow_sccm": q,
        "molar_flow_mol_s": molar_flow,
        "particle_flow_s-1": particle_flow,
        "integrated_moles_mol": moles,
        "integrated_particles": particles,
        "reference_pressure_Pa": float(reference_pressure_Pa),
        "reference_temperature_K": float(reference_temperature_K),
        "method": "ideal-gas standard-volume conversion and trapezoidal time integration",
        "interpretation": "Gas-specific flowmeter calibration and the stated standard reference conditions are required for physical accuracy.",
    }


def effective_pumping_speed_m3_s(throughput_Pa_m3_s: Any, pressure_Pa: Any) -> np.ndarray:
    """Compute S = Q/p using throughput in Pa·m³/s and pressure in Pa."""
    q, p = np.broadcast_arrays(
        np.asarray(throughput_Pa_m3_s, dtype=float),
        np.asarray(pressure_Pa, dtype=float),
    )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
        raise ValueError("Throughput and pressure must be finite.")
    if np.any(q < 0) or np.any(p <= 0):
        raise ValueError("Throughput must be nonnegative and pressure must be positive.")
    return q / p
