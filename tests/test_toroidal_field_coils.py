import numpy as np
import pytest
from tokamak_analysis.toroidal_field_coils import analyze_toroidal_field_coil, integrate_pickup_coil_voltage

def test_current_conversion_and_uncertainty():
    r=analyze_toroidal_field_coil([0,.1,.2],[100,110,120],calibration_T_per_unit=.01,
      signal_kind="current",baseline=100,signal_uncertainty=[1,1,1],
      calibration_uncertainty_T_per_unit=.001)
    assert r["B_toroidal_T"].tolist()==pytest.approx([0,.1,.2])
    assert r["peak_to_peak_B_toroidal_T"]==pytest.approx(.2)
    assert r["sampling_rate_Hz"]==pytest.approx(10)
    assert r["B_toroidal_uncertainty_T"][1]==pytest.approx(np.hypot(.01,.01))

@pytest.mark.parametrize("t,x", [([0,0],[1,2]),([0,1],[1,np.nan]),([0],[1])])
def test_invalid_inputs(t,x):
    with pytest.raises(ValueError):
        analyze_toroidal_field_coil(t,x,calibration_T_per_unit=1)

def test_voltage_is_not_automatically_integrated():
    r=analyze_toroidal_field_coil([0,1],[1,-1],calibration_T_per_unit=.2,signal_kind="voltage")
    assert r["B_toroidal_T"].tolist()==pytest.approx([.2,-.2])
    assert any("integration" in x.lower() for x in r["limitations"])

def test_field_mode_requires_identity_calibration():
    with pytest.raises(ValueError):
        analyze_toroidal_field_coil([0,1],[1,2],calibration_T_per_unit=2,signal_kind="field")


def test_pickup_voltage_integration_recovers_field_change():
    # V = S*dB/dt; constant 2 T/s over 1 s yields a 2 T field change.
    r = integrate_pickup_coil_voltage(
        [0, 0.5, 1.0], [4, 4, 4],
        sensitivity_V_per_T_per_s=2, initial_field_T=3,
        voltage_uncertainty_V=[0.1, 0.1, 0.1],
        sensitivity_uncertainty_V_per_T_per_s=0.1,
        initial_field_uncertainty_T=0.05,
    )
    assert r["dB_dt_T_s"].tolist() == pytest.approx([2, 2, 2])
    assert r["delta_B_T"].tolist() == pytest.approx([0, 1, 2])
    assert r["B_toroidal_T"].tolist() == pytest.approx([3, 4, 5])
    assert r["B_toroidal_uncertainty_T"][-1] > 0
    assert "initial-field boundary condition" in r["limitations"][1]

@pytest.mark.parametrize("kwargs", [
    {"sensitivity_V_per_T_per_s": 0, "initial_field_T": 0},
    {"sensitivity_V_per_T_per_s": 1, "initial_field_T": float("nan")},
])
def test_pickup_voltage_integration_rejects_invalid_calibration(kwargs):
    with pytest.raises(ValueError):
        integrate_pickup_coil_voltage([0, 1], [1, 1], **kwargs)

def test_pickup_voltage_integration_rejects_bad_uncertainty():
    with pytest.raises(ValueError):
        integrate_pickup_coil_voltage(
            [0, 1], [1, 1], sensitivity_V_per_T_per_s=1,
            initial_field_T=0, voltage_uncertainty_V=[0, -1]
        )

def test_pickup_voltage_uncertainty_accounts_for_shared_trapezoid_samples():
    # Unequal time steps and independent pointwise noise. The interior samples
    # contribute to two adjacent trapezoids, so interval variances alone
    # underestimate cumulative uncertainty.
    time = np.array([0.0, 0.1, 0.3, 0.6])
    sigma_v = np.array([0.1, 0.2, 0.3, 0.4])
    sensitivity = 2.0
    result = integrate_pickup_coil_voltage(
        time, np.zeros_like(time),
        sensitivity_V_per_T_per_s=sensitivity,
        initial_field_T=0.0,
        voltage_uncertainty_V=sigma_v,
    )
    q = np.diff(time) / (2.0 * sensitivity)
    expected_weights = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [q[0], q[0], 0.0, 0.0],
        [q[0], q[0] + q[1], q[1], 0.0],
        [q[0], q[0] + q[1], q[1] + q[2], q[2]],
    ])
    expected = np.sqrt(np.sum((expected_weights * sigma_v) ** 2, axis=1))
    np.testing.assert_allclose(
        result["B_toroidal_uncertainty_T"], expected, rtol=1e-13, atol=0.0
    )



def test_registered_analysis_integrates_pickup_voltage_and_propagates_uncertainty():
    r = analyze_toroidal_field_coil(
        [0.0, 0.5, 1.0], [0.0, 0.0, 0.0],
        calibration_T_per_unit=1.0, signal_kind="field",
        pickup_voltage_V=[4.0, 4.0, 4.0],
        pickup_sensitivity_V_per_T_per_s=2.0,
        initial_field_T=3.0,
        pickup_voltage_uncertainty_V=[0.1, 0.1, 0.1],
        pickup_sensitivity_uncertainty_V_per_T_per_s=0.1,
        initial_field_uncertainty_T=0.05,
    )
    assert r["B_toroidal_integrated_T"].tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert r["delta_B_T"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert r["dB_dt_T_s"].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert r["B_toroidal_integrated_uncertainty_T"][-1] > 0


def test_registered_analysis_requires_pickup_integration_boundary_conditions():
    with pytest.raises(ValueError, match="sensitivity.*initial_field"):
        analyze_toroidal_field_coil(
            [0.0, 1.0], [0.0, 0.0], calibration_T_per_unit=1.0,
            pickup_voltage_V=[1.0, 1.0],
        )
