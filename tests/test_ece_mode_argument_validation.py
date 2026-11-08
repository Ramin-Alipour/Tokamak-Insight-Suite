import numpy as np
import pytest

from tokamak_analysis.ece_analysis import analyze_ece


def test_ece_rejects_optical_depth_outside_brightness_temperature_mode():
    with pytest.raises(ValueError, match="only consumed in brightness_temperature"):
        analyze_ece([80e9, 81e9], mode="resonance_mapping", optical_depth=1.0)


def test_ece_rejects_noise_model_outside_channel_fluctuations_mode():
    with pytest.raises(ValueError, match="only consumed in channel_fluctuations"):
        analyze_ece([80e9, 81e9], mode="brightness_temperature", noise_std=0.1)


def test_ece_rejects_optical_depth_in_channel_fluctuations_mode():
    with pytest.raises(ValueError, match="only consumed in brightness_temperature"):
        analyze_ece(
            [80e9, 81e9],
            mode="channel_fluctuations",
            signal=np.ones((8, 2)),
            time_s=np.arange(8.0),
            optical_depth=1.0,
        )
