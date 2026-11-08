from __future__ import annotations

import numpy as np

from tis.products.langmuir import (
    analyze_ball_pen,
    analyze_gundestrup,
    analyze_langmuir_iv,
    analyze_multipin,
    analyze_potential_arrays,
    analyze_probe_iv_compatibility,
)


def test_langmuir_product_named_facades_execute_their_canonical_solvers():
    # A smooth Maxwellian-like retarding branch plus ion-current offset.
    voltage = np.linspace(-8.0, 8.0, 81)
    current = 2.0e-4 + 1.0e-3 * np.exp((voltage - 8.0) / 2.0)
    result = analyze_langmuir_iv(
        voltage, current, electron_fit_range=(-3.0, 5.0),
        ion_fit_range=(-8.0, -5.0),
    )
    assert result["electron_temperature_eV"] is not None
    assert result["electron_temperature_eV"] > 0.0

    compatibility = analyze_probe_iv_compatibility(
        voltage[:6], current[:6]
    )
    assert compatibility["analysis_scope"] == "descriptive_short_sweep_compatibility_only"


def test_ball_pen_gundestrup_and_potential_array_facades():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    ball_pen = analyze_ball_pen(time, [1.0, 1.1, 0.9, 1.0], [0.5, 0.6, 0.4, 0.5])
    assert ball_pen["potential_difference_V"].shape == (4,)

    angles = np.linspace(0.0, 2.0*np.pi, 12, endpoint=False)
    currents = 2.0 + 0.3*np.cos(angles) - 0.2*np.sin(angles)
    gundestrup = analyze_gundestrup(angles, currents)
    assert np.isclose(gundestrup["first_harmonic_amplitude_A"], np.hypot(0.3, -0.2))

    potential = np.array([4.0, 3.0, 2.0])
    floating = np.array([1.0, 1.0, 1.0])
    arrays = analyze_potential_arrays(
        potential, floating, coordinate_m=np.array([0.0, 0.5, 1.0])
    )
    np.testing.assert_allclose(arrays["electric_field_V_m"], [2.0, 2.0, 2.0])


def test_multipin_product_facade_exposes_voltage_covariance_propagation():
    result = analyze_multipin(
        [0.0, 1.0], [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]],
        [0.0, 0.5, 1.0], potential_covariance_V2=np.eye(3)*1.0e-4,
    )
    assert result["electric_field_standard_uncertainty_V_m"].shape == (2, 3)
    assert result["uncertainty_scope"] == "shared_pin_covariance"



def test_probe_registry_labels_and_contracts_match_their_physical_solver_modules():
    from tis.products.langmuir import run_probe_analysis
    from tis.products.catalog import list_product_analyses

    entries = {entry["id"]: entry for entry in list_product_analyses(
        "langmuir-electrostatic-probes"
    )}
    assert entries[16]["module"] == "probe_analysis"
    assert "compatibility" in entries[16]["name"].lower()
    assert entries[17]["module"] == "ball_pen_probe"
    assert "ball-pen" in entries[17]["name"].lower()
    assert entries[18]["module"] == "gundestrup_probe"
    assert "gundestrup" in entries[18]["name"].lower()
    assert entries[19]["module"] == "multipin_probe"
    assert "potential_covariance_V2" in {
        item["name"] for item in entries[19]["execution_contract"]["input_specs"]
    }
    assert entries[20]["module"] == "potential_analysis"
    assert callable(run_probe_analysis)
