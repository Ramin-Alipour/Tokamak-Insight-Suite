import numpy as np

from tokamak_analysis.cts_em_geometry import propagate_cold_plasma_mode_coupling


def test_degenerate_subspace_power_is_reported_separately_from_total_power():
    operators = np.repeat(np.diag([1.0, 1.0, 3.0])[None, :, :], 4, axis=0)
    path = np.linspace(0.0, 0.3, 4)
    beta = np.zeros((4, 3))
    initial = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])

    result = propagate_cold_plasma_mode_coupling(
        operators, path, beta, initial
    )

    np.testing.assert_allclose(result["total_mode_power"], 14.0, rtol=1e-12)
    np.testing.assert_allclose(result["degenerate_subspace_power"], 5.0, rtol=1e-12)
    for sample_groups in result["degenerate_subspace_groups"]:
        assert len(sample_groups) == 1
        assert sample_groups[0]["mode_indices"] == [0, 1]
        assert sample_groups[0]["power"] == pytest.approx(5.0, rel=1e-12)
    assert np.max(result["degenerate_subspace_power"]) < np.max(
        result["total_mode_power"]
    )


def test_non_degenerate_modes_have_zero_degenerate_subspace_power():
    operators = np.repeat(np.diag([1.0, 2.0, 3.0])[None, :, :], 3, axis=0)
    result = propagate_cold_plasma_mode_coupling(
        operators,
        np.array([0.0, 0.1, 0.2]),
        np.zeros((3, 3)),
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
    )

    np.testing.assert_array_equal(result["degenerate_subspace_power"], 0.0)
    assert result["degenerate_subspace_groups"] == [[], [], []]
    assert np.all(~result["degenerate_subspace_detected"])
