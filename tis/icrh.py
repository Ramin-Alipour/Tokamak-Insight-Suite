from __future__ import annotations
import numpy as np

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def analyze_icrh_power(launched_power_W, *, magnetic_field_T: float, ion_mass_amu: float, charge_state: int = 1, harmonic: int = 1, transmission_fraction: float = 1.0, reflection_fraction: float = 0.0, absorption_fraction: float = 1.0) -> dict:
    """Estimate ICRH absorbed power and ion-cyclotron resonance frequency."""
    power = np.asarray(launched_power_W, dtype=float)
    if power.size == 0 or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError('launched_power_W must be finite and nonnegative.')
    if not np.isfinite(magnetic_field_T) or magnetic_field_T <= 0 or not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0:
        raise ValueError('Magnetic field and ion mass must be positive and finite.')
    for name, value in [('transmission_fraction', transmission_fraction), ('reflection_fraction', reflection_fraction), ('absorption_fraction', absorption_fraction)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between zero and one.')
    if transmission_fraction + reflection_fraction > 1 + 1e-12:
        raise ValueError('Transmission plus reflection fractions cannot exceed one.')
    if isinstance(charge_state, bool) or not isinstance(charge_state, int) or charge_state < 1 or isinstance(harmonic, bool) or not isinstance(harmonic, int) or harmonic < 1:
        raise ValueError('charge_state and harmonic must be positive integers.')
    fci = harmonic*charge_state*E_CHARGE*magnetic_field_T/(2*np.pi*ion_mass_amu*AMU)
    coupled = power*transmission_fraction*(1-reflection_fraction)
    absorbed = coupled*absorption_fraction
    return {'diagnostic': 'ICRH power', 'launched_power_W': power, 'coupled_power_W': coupled,
        'absorbed_power_W': absorbed, 'total_absorbed_power_W': float(np.sum(absorbed)),
        'ion_cyclotron_resonance_frequency_Hz': float(fci), 'magnetic_field_T': float(magnetic_field_T),
        'ion_mass_amu': float(ion_mass_amu), 'charge_state': charge_state, 'harmonic': harmonic,
        'assumptions_and_limits': ['Power fractions must be supplied from launcher/coupling measurements or modeling.',
            'Resonance frequency is a non-relativistic single-particle estimate.',
            'Wave propagation, mode conversion, minority heating and deposition profiles are not modeled.']}
