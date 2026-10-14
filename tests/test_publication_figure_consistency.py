"""Figure-derived consistency check against the IR-T1 hard-X-ray paper.

The five energies below are digitized from Fig. 6, 0–5 ms panel. They are
test-only figure-derived values, not the original detector event file. Table 1
reports 5 counts and total energy 3301 keV for the same window.
"""
import numpy as np
import pytest

from tis.core import xray_spectrum_summary


def test_hard_xray_figure_digitization_reproduces_table_mean_energy():
    digitized_event_energies_keV = np.array([
        318.68, 620.88, 673.08, 750.00, 925.82
    ])
    result = xray_spectrum_summary(
        digitized_event_energies_keV,
        np.ones(len(digitized_event_energies_keV)),
        bin_representation="counts_per_bin",
    )
    table_mean_energy_keV = 3301.0 / 5.0
    assert result["moments_valid"] is True
    assert result["integrated_counts_keV_weighted"] == pytest.approx(5.0)
    # Figure digitization is pixel-limited; allow 1% against the table value.
    assert result["mean_energy_keV"] == pytest.approx(table_mean_energy_keV, rel=0.01)
    assert result["mean_energy_keV"] == pytest.approx(657.69, rel=0.002)
