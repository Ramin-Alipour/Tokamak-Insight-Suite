from __future__ import annotations
import numpy as np

PA_PER_UNIT = {'Pa': 1.0, 'mbar': 100.0, 'Torr': 133.32236842105263, 'kPa': 1000.0, 'bar': 100000.0}


def analyze_pressure_gauge(reading, *, calibration_Pa_per_unit: float, offset_Pa: float = 0.0, reading_uncertainty: float | None = None, calibration_relative_uncertainty: float | None = None, valid_range_Pa=None) -> dict:
    """Calibrate pressure-gauge readings into SI pressure and common vacuum units."""
    x = np.asarray(reading, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError('reading must be finite and nonempty.')
    if not np.isfinite(calibration_Pa_per_unit) or calibration_Pa_per_unit == 0 or not np.isfinite(offset_Pa):
        raise ValueError('Calibration slope must be finite and nonzero; offset finite.')
    for name, value in [('reading_uncertainty', reading_uncertainty), ('calibration_relative_uncertainty', calibration_relative_uncertainty)]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative.')
    pressure = x*calibration_Pa_per_unit + offset_Pa
    if np.any(pressure < 0):
        raise ValueError('Calibrated absolute pressure is negative; check calibration and gauge convention.')
    uncertainty = None
    if reading_uncertainty is not None or calibration_relative_uncertainty is not None:
        variance = np.zeros_like(pressure)
        if reading_uncertainty is not None:
            variance += (calibration_Pa_per_unit*reading_uncertainty)**2
        if calibration_relative_uncertainty is not None:
            variance += (x*calibration_Pa_per_unit*calibration_relative_uncertainty)**2
        uncertainty = np.sqrt(variance)
    in_range = None
    if valid_range_Pa is not None:
        if len(valid_range_Pa) != 2 or not np.all(np.isfinite(valid_range_Pa)) or valid_range_Pa[0] < 0 or valid_range_Pa[0] >= valid_range_Pa[1]:
            raise ValueError('valid_range_Pa must be a finite increasing nonnegative pair.')
        in_range = (pressure >= valid_range_Pa[0]) & (pressure <= valid_range_Pa[1])
    return {'diagnostic': 'Pressure gauges', 'pressure_Pa': pressure,
        'pressure_mbar': pressure/100.0, 'pressure_Torr': pressure/PA_PER_UNIT['Torr'],
        'pressure_standard_uncertainty_Pa': uncertainty, 'within_valid_range': in_range,
        'calibration_Pa_per_unit': float(calibration_Pa_per_unit), 'offset_Pa': float(offset_Pa),
        'assumptions_and_limits': ['Calibration must be specific to the gauge type, gas and operating range.',
            'Gas species corrections and conductance between gauge and plasma vessel are not inferred.',
            'Negative calibrated absolute pressure is rejected.']}
