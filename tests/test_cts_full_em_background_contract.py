import numpy as np
import pytest

from tokamak_analysis.cts_full_em import _physical_background_zero_state


def test_omitted_physical_background_is_unknown_not_assumed_zero():
    zero, status = _physical_background_zero_state()
    assert zero is None
    assert status == "background_not_supplied_not_assumed_zero"


def test_explicit_zero_receiver_grid_background_is_reported_as_zero():
    zero, status = _physical_background_zero_state(
        supplied_spectrum=np.zeros(5, dtype=float)
    )
    assert zero is True
    assert status == "zero_background_exact_zero_receiver_grid_spectrum"


def test_nonzero_receiver_grid_background_is_not_reported_as_zero():
    spectrum = np.array([0.0, 1e-18, 0.0])
    zero, status = _physical_background_zero_state(supplied_spectrum=spectrum)
    assert zero is False
    assert status == "nonzero_fixed_receiver_grid_background"


def test_radiative_transfer_zero_background_status_is_preserved():
    zero, status = _physical_background_zero_state(
        transfer_diagnostics={
            "zero_background": True,
            "background_status": "zero_background_exact_zero_radiance",
        }
    )
    assert zero is True
    assert status == "zero_background_exact_zero_radiance"


def test_radiative_transfer_incident_only_state_is_not_zero():
    zero, status = _physical_background_zero_state(
        transfer_diagnostics={
            "zero_background": False,
            "background_status": "incident_radiance_only_no_plasma_background",
        }
    )
    assert zero is False
    assert status == "incident_radiance_only_no_plasma_background"


def test_incomplete_radiative_transfer_status_is_rejected():
    with pytest.raises(ValueError, match="declare boolean zero_background"):
        _physical_background_zero_state(
            transfer_diagnostics={"background_status": "unknown"}
        )


def test_background_state_inputs_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _physical_background_zero_state(
            transfer_diagnostics={
                "zero_background": True,
                "background_status": "zero_background_exact_zero_radiance",
            },
            supplied_spectrum=np.zeros(3),
        )
