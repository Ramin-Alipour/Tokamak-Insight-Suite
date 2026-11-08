"""Public execution API for Langmuir and electrostatic probe workflows."""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "langmuir-electrostatic-probes"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a registered workflow owned by the probe product."""
    return run_analysis(analysis_id, *args, **kwargs)


def run_probe_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a registered Langmuir/electrostatic probe workflow (IDs 2, 16–20)."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_langmuir_iv(voltage_V, current_A, **kwargs: Any) -> Any:
    """Run the calibrated Langmuir I-V fit and optional sheath density inference."""
    return run_probe_analysis(2, voltage_V, current_A, **kwargs)


def analyze_probe_iv_compatibility(voltage_V, current_A, **kwargs: Any) -> Any:
    """Run the legacy-compatible probe I-V adapter (short sweeps are descriptive)."""
    return run_probe_analysis(16, voltage_V, current_A, **kwargs)


def analyze_ball_pen(time_s, ball_pen_V, floating_V=None) -> Any:
    """Summarize synchronized ball-pen and optional floating-potential signals."""
    return run_probe_analysis(17, time_s, ball_pen_V, floating_V)


def analyze_gundestrup(angles_rad, collector_currents_A) -> Any:
    """Fit the first angular harmonic of directional collector currents."""
    return run_probe_analysis(18, angles_rad, collector_currents_A)


def analyze_multipin(time_s, potentials_V, positions_m, **kwargs: Any) -> Any:
    """Infer the array-axis electric field and propagate optional pin covariance."""
    return run_probe_analysis(19, time_s, potentials_V, positions_m, **kwargs)


def analyze_potential_arrays(potential_V, floating_potential_V, **kwargs: Any) -> Any:
    """Compare plasma/floating potential arrays and optionally infer E=-dVp/dx."""
    return run_probe_analysis(20, potential_V, floating_potential_V, **kwargs)
