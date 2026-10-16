from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .core import reflectometry_cutoff_density


def trace_o_mode_ray(x_m, electron_density_m3, *, frequency_hz: float, propagation_length_m: float, initial_x_m: float, initial_slope: float = 0.0, samples: int = 201) -> dict:
    """Trace a 2-D slab O-mode ray through a measured density profile."""
    x = np.asarray(x_m, dtype=float).ravel()
    ne = np.asarray(electron_density_m3, dtype=float).ravel()
    if x.size < 5 or x.size != ne.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(ne)):
        raise ValueError('Position and density profiles must be finite, equal length and have at least five points.')
    if np.any(np.diff(x) <= 0) or np.any(ne < 0):
        raise ValueError('Position must increase strictly and density must be nonnegative.')
    if not np.isfinite(frequency_hz) or frequency_hz <= 0 or not np.isfinite(propagation_length_m) or propagation_length_m <= 0:
        raise ValueError('Frequency and propagation length must be positive and finite.')
    if not np.isfinite(initial_x_m) or not np.isfinite(initial_slope) or not x[0] <= initial_x_m <= x[-1]:
        raise ValueError('Initial ray position must lie inside the profile and slope must be finite.')
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 10:
        raise ValueError('samples must be an integer of at least 10.')
    ncrit = reflectometry_cutoff_density(frequency_hz, mode='O')['cutoff_density_m-3']
    refractive_sq = 1 - ne / ncrit
    if np.any(refractive_sq <= 0):
        raise ValueError('Profile reaches or exceeds the O-mode cutoff; this propagator cannot cross an evanescent region.')
    refractive_index = np.sqrt(refractive_sq)
    n_fun = interp1d(x, refractive_index, kind='linear', bounds_error=True)
    dn_dx = np.gradient(refractive_index, x, edge_order=2)
    dn_fun = interp1d(x, dn_dx, kind='linear', bounds_error=True)
    def rhs(z, state):
        xpos, slope = state
        nloc = float(n_fun(xpos))
        return [slope, float(dn_fun(xpos)) / nloc * (1 + slope*slope)]
    z_eval = np.linspace(0, propagation_length_m, samples)
    solution = solve_ivp(rhs, (0, propagation_length_m), [initial_x_m, initial_slope], t_eval=z_eval, rtol=1e-8, atol=1e-10)
    if not solution.success:
        raise ValueError(f'Ray integration failed: {solution.message}')
    return {'diagnostic': 'Refractive-index / ray-tracing diagnostics', 'propagation_coordinate_m': solution.t,
        'ray_position_m': solution.y[0], 'ray_slope_dx_dz': solution.y[1],
        'initial_angle_rad': float(np.arctan(initial_slope)), 'final_angle_rad': float(np.arctan(solution.y[1, -1])),
        'angular_deflection_rad': float(np.arctan(solution.y[1, -1])-np.arctan(initial_slope)),
        'O_mode_cutoff_density_m-3': float(ncrit), 'refractive_index_profile': refractive_index,
        'assumptions_and_limits': ['2-D slab, geometric-optics, cold collisionless O-mode approximation.',
            'No diffraction, finite beam width, mode conversion, absorption or equilibrium toroidal geometry.',
            'Ray is rejected if the supplied profile reaches cutoff; turning-point handling is not implemented.']}
