import numpy as np
import pytest

from tokamak_analysis import cts_full_em


def test_full_em_convergence_reports_paired_harmonic_quadrature_refinement(monkeypatch):
    def fake_forward(frequency, probe_frequency, geometry, *, harmonic_order, quadrature_order, **kwargs):
        return {"spectral_density": np.array([1.0, 2.0, 3.0]) + 1.0 / (harmonic_order * quadrature_order)}

    monkeypatch.setattr(cts_full_em, "full_em_cts_spectral_density", fake_forward)
    result = cts_full_em.assess_full_em_cts_numerical_convergence(
        np.array([-1.0, 0.0, 1.0]), 1.0e9, {},
        harmonic_orders=(4, 8, 12), quadrature_orders=(4, 8, 12),
        relative_tolerance=1e-2,
    )
    assert len(result["levels"]) == 3
    assert result["levels"][0]["relative_l2_change_from_previous"] is None
    assert result["levels"][-1]["relative_l2_change_from_previous"] >= 0.0
    assert result["convergence_status"] in {
        "converged_at_final_refinement", "not_converged_at_final_refinement"
    }
    assert result["final_spectrum"].shape == (3,)


def test_full_em_convergence_requires_paired_increasing_refinement_levels():
    with pytest.raises(ValueError, match="paired 1-D sequences"):
        cts_full_em.assess_full_em_cts_numerical_convergence(
            np.array([-1.0, 0.0, 1.0]), 1.0e9, {},
            harmonic_orders=(4, 8), quadrature_orders=(4,),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        cts_full_em.assess_full_em_cts_numerical_convergence(
            np.array([-1.0, 0.0, 1.0]), 1.0e9, {},
            harmonic_orders=(4, 4), quadrature_orders=(4, 8),
        )


def test_full_em_inversion_enables_automatic_convergence_by_default():
    import inspect

    parameters = inspect.signature(cts_full_em.analyze_full_em_cts_spectrum).parameters
    assert parameters["numerical_convergence"].default is True
    assert parameters["convergence_harmonic_orders"].default is None
    assert parameters["convergence_quadrature_orders"].default is None
    assert parameters["convergence_relative_tolerance"].default == pytest.approx(1e-3)
