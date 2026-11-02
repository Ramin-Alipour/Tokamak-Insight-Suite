"""Tests for geometry-covariance path handling in full-EM CTS."""
import numpy as np
import pytest

from tokamak_analysis.cts_full_em import (
    _geometry_path_tokens,
    _set_geometry_path,
)


def test_geometry_path_tokens_support_nested_mapping_and_sequence_paths():
    assert _geometry_path_tokens(
        "frequency_resolved_geometry[0].scattering_samples[1].scattering_location_m[2]"
    ) == [
        "frequency_resolved_geometry", 0, "scattering_samples", 1,
        "scattering_location_m", 2,
    ]


def test_set_geometry_path_updates_only_requested_scalar():
    geometry = {
        "frequency_resolved_geometry": [{
            "scattering_samples": [{"scattering_location_m": [1.0, 2.0, 3.0]}]
        }]
    }
    _set_geometry_path(
        geometry,
        "frequency_resolved_geometry[0].scattering_samples[0].scattering_location_m[1]",
        0.25,
    )
    assert geometry["frequency_resolved_geometry"][0]["scattering_samples"][0][
        "scattering_location_m"
    ] == [1.0, 0.25, 3.0]


@pytest.mark.parametrize("path", ["", "a..b", "a[", "a[-1]", ".a", "a."])
def test_geometry_path_rejects_malformed_paths(path):
    with pytest.raises(ValueError):
        _geometry_path_tokens(path)
