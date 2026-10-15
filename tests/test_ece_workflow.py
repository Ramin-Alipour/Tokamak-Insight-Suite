import numpy as np
import pytest

from tis.ece import analyze_ece_channels


def test_ece_fundamental_resonance_and_major_radius_mapping():
    # 28 GHz is approximately the fundamental electron-cyclotron frequency at 1 T.
    result = analyze_ece_channels(
        [28e9], toroidal_field_axis_T=2.0, major_radius_axis_m=1.0
    )
    assert result["resonance_field_T"][0] == pytest.approx(1.001, rel=0.01)
    assert result["approximate_major_radius_m"][0] == pytest.approx(1.998, rel=0.02)


def test_ece_optical_depth_temperature_correction():
    result = analyze_ece_channels(
        [60e9, 70e9], toroidal_field_axis_T=2.0, major_radius_axis_m=1.0,
        brightness_temperature_eV=[100, 200], optical_depth=[np.log(2), np.log(2)],
    )
    np.testing.assert_allclose(result["electron_temperature_eV"], [200, 400])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"toroidal_field_axis_T": 0, "major_radius_axis_m": 1},
        {"toroidal_field_axis_T": 2, "major_radius_axis_m": -1},
        {"toroidal_field_axis_T": 2, "major_radius_axis_m": 1, "harmonic": 0},
    ],
)
def test_ece_rejects_invalid_physical_inputs(kwargs):
    with pytest.raises(ValueError):
        analyze_ece_channels([60e9], **kwargs)


def test_ece_rejects_mismatched_brightness_channels():
    with pytest.raises(ValueError, match="match frequency_hz"):
        analyze_ece_channels(
            [60e9, 70e9], toroidal_field_axis_T=2, major_radius_axis_m=1,
            brightness_temperature_eV=[100],
        )
