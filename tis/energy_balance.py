import numpy as np


def analyze_energy_balance(time_s, stored_energy_J, input_power_W, *, radiated_power_W=None, transport_loss_power_W=None) -> dict:
    """Compute stored-energy derivative, power balance and confinement time."""
    t = np.asarray(time_s, dtype=float).ravel()
    w = np.asarray(stored_energy_J, dtype=float).ravel()
    pin = np.asarray(input_power_W, dtype=float).ravel()
    if t.size < 3 or len({t.size, w.size, pin.size}) != 1 or not all(np.all(np.isfinite(a)) for a in (t, w, pin)):
        raise ValueError('Inputs must be finite equal-length profiles with at least three samples.')
    if np.any(np.diff(t) <= 0) or np.any(w < 0) or np.any(pin < 0):
        raise ValueError('Time must increase and energy/power must be nonnegative.')
    dwdt = np.gradient(w, t, edge_order=2)
    prad = np.zeros_like(pin) if radiated_power_W is None else np.asarray(radiated_power_W, dtype=float).ravel()
    if prad.size != t.size or not np.all(np.isfinite(prad)) or np.any(prad < 0):
        raise ValueError('Radiated power must be finite, nonnegative and match time.')
    inferred = pin - prad - dwdt
    supplied = None if transport_loss_power_W is None else np.asarray(transport_loss_power_W, dtype=float).ravel()
    residual = None
    if supplied is not None:
        if supplied.size != t.size or not np.all(np.isfinite(supplied)) or np.any(supplied < 0):
            raise ValueError('Transport loss must be finite, nonnegative and match time.')
        residual = dwdt - (pin - prad - supplied)
    loss = pin - dwdt
    tau = np.divide(w, loss, out=np.full_like(w, np.nan), where=loss > 0)
    fraction = np.divide(prad, pin, out=np.full_like(pin, np.nan), where=pin > 0)
    return {'diagnostic': 'Energy balance', 'time_s': t, 'stored_energy_J': w, 'dWdt_W': dwdt,
        'input_power_W': pin, 'radiated_power_W': prad, 'inferred_transport_loss_power_W': inferred,
        'supplied_transport_loss_power_W': supplied, 'energy_balance_residual_W': residual,
        'total_loss_power_from_balance_W': loss, 'energy_confinement_time_s': tau,
        'radiated_power_fraction': fraction,
        'assumptions_and_limits': ['Power terms must use consistent time alignment and definitions.',
            'Numerical differentiation amplifies measurement noise.',
            'Negative inferred transport loss flags inconsistent inputs or missing power terms.']}
