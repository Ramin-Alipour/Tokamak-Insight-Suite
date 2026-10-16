import numpy as np
from tis.energy_balance import analyze_energy_balance


def test_energy_balance_steady_state_confinement():
    t = np.linspace(0, 1, 11)
    w = np.full(11, 100.0)
    pin = np.full(11, 50.0)
    r = analyze_energy_balance(t, w, pin, radiated_power_W=np.full(11, 10.0), transport_loss_power_W=np.full(11, 40.0))
    assert np.max(np.abs(r['energy_balance_residual_W'])) < 1e-8
    np.testing.assert_allclose(r['energy_confinement_time_s'], 2.0)
