import numpy as np


def analyze_strike_point_tracking(surface_maps, time_s, R_map_m, Z_map_m) -> dict:
    """Track peak-emission/heat-flux locations across calibrated surface maps."""
    data = np.asarray(surface_maps, dtype=float)
    t = np.asarray(time_s, dtype=float).ravel()
    R = np.asarray(R_map_m, dtype=float)
    Z = np.asarray(Z_map_m, dtype=float)
    if data.ndim != 3 or data.shape[0] != t.size or data.shape[1:] != R.shape or R.shape != Z.shape:
        raise ValueError('Maps must be time-by-y-by-x and R/Z coordinate grids must match each frame.')
    if data.shape[0] < 2 or min(data.shape[1:]) < 2 or not all(np.all(np.isfinite(a)) for a in (data, t, R, Z)) or np.any(np.diff(t) <= 0):
        raise ValueError('Maps and coordinates must be finite, with at least two frames and increasing time.')
    indices = np.argmax(data.reshape(data.shape[0], -1), axis=1)
    rows, cols = np.unravel_index(indices, data.shape[1:])
    r_peak, z_peak = R[rows, cols], Z[rows, cols]
    speed = np.hypot(np.gradient(r_peak, t), np.gradient(z_peak, t))
    return {'diagnostic': 'Strike-point / hot-spot tracking', 'time_s': t,
        'peak_map_value': data.reshape(data.shape[0], -1).max(axis=1),
        'peak_pixel_row': rows, 'peak_pixel_column': cols,
        'peak_R_m': r_peak, 'peak_Z_m': z_peak, 'apparent_peak_speed_m_s': speed,
        'assumptions_and_limits': ['Coordinate maps must be registered to the target/divertor surface.',
            'Maximum-pixel tracking can jump between competing hot spots and is sensitive to noise/saturation.',
            'Apparent peak motion is not automatically strike-point motion unless the measured map represents strike-point emission/heat flux.']}
