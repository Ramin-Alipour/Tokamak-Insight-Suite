from __future__ import annotations
import numpy as np

K_B = 1.380649e-23
STANDARD_PRESSURE_PA = 101325.0
STANDARD_TEMPERATURE_K = 273.15
SCCM_TO_M3_S = 1e-6 / 60.0


def analyze_gas_flow_meter(reading, *, calibration_sccm_per_unit: float, offset_sccm: float = 0.0, standard_pressure_Pa: float = STANDARD_PRESSURE_PA, standard_temperature_K: float = STANDARD_TEMPERATURE_K) -> dict:
    """Calibrate gas flow in sccm and convert to standard throughput and particle rate."""
    x = np.asarray(reading, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError('reading must be finite and nonempty.')
    if not np.isfinite(calibration_sccm_per_unit) or calibration_sccm_per_unit == 0 or not np.isfinite(offset_sccm):
        raise ValueError('Calibration slope must be finite and nonzero; offset finite.')
    if not np.isfinite(standard_pressure_Pa) or standard_pressure_Pa <= 0 or not np.isfinite(standard_temperature_K) or standard_temperature_K <= 0:
        raise ValueError('Standard pressure and temperature must be positive and finite.')
    flow_sccm = x*calibration_sccm_per_unit + offset_sccm
    if np.any(flow_sccm < 0):
        raise ValueError('Calibrated gas flow is negative; check calibration and flow direction.')
    flow_m3_s = flow_sccm*SCCM_TO_M3_S
    throughput_Pa_m3_s = standard_pressure_Pa*flow_m3_s
    particle_rate_s = throughput_Pa_m3_s/(K_B*standard_temperature_K)
    return {'diagnostic': 'Gas flow meters', 'flow_sccm': flow_sccm,
        'standard_volumetric_flow_m3_s': flow_m3_s, 'gas_throughput_Pa_m3_s': throughput_Pa_m3_s,
        'particle_rate_s-1': particle_rate_s, 'standard_pressure_Pa': float(standard_pressure_Pa),
        'standard_temperature_K': float(standard_temperature_K),
        'assumptions_and_limits': ['sccm conversion assumes the declared standard pressure and temperature.',
            'Gas-specific meter correction factors and mixture composition must be supplied externally.',
            'Throughput and particle rate represent standard-condition equivalents, not local vessel volumetric flow.']}
