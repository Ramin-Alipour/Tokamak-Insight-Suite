import numpy as np
import pytest
import tokamak_analysis.ece_kinetic as ece
import tis.physics.shared.ece_kinetic as ece_physics


def test_profile_solver_builds_frequency_segment_absorption_and_transfer(monkeypatch):
    def fake_absorption(frequency, wavevector, magnetic_field, **kwargs):
        return {"absorption_coefficient_m_inverse": frequency * 1e-12 + kwargs["electron_density_m3"] * 1e-30}
    monkeypatch.setattr(ece_physics, "kinetic_ece_absorption_coefficient", fake_absorption)
    out = ece.solve_kinetic_ece_profile(
        [80e9, 100e9], [0.2, 0.3],
        wavevector_rad_m=np.array([[1.0, 0, 0], [1.1, 0, 0]]),
        magnetic_field_T=np.array([[0, 0, 2.0], [0, 0, 2.1]]),
        polarization=np.array([[1, 0, 0], [1, 0, 0]], dtype=complex),
        electron_density_m3=[1e19, 2e19],
        electron_temperature_eV=[500, 700],
    )
    assert out["absorption_coefficient_m_inverse"].shape == (2, 2)
    assert out["brightness_temperature_out_K"].shape == (2,)
    assert out["local_mode_absorption"][1][1]["absorption_coefficient_m_inverse"] == pytest.approx(0.1)


def test_profile_solver_rejects_incomplete_mode_geometry():
    with pytest.raises(ValueError, match="wavevector_rad_m"):
        ece.solve_kinetic_ece_profile(
            90e9, [1.0], wavevector_rad_m=[[1, 0]],
            magnetic_field_T=[[0, 0, 2]], polarization=[[1, 0, 0]],
            electron_density_m3=[1e19], electron_temperature_eV=[100],
        )


def test_profile_solver_does_not_hide_kinetic_gain(monkeypatch):
    monkeypatch.setattr(
        ece_physics, "kinetic_ece_absorption_coefficient",
        lambda *args, **kwargs: {"absorption_coefficient_m_inverse": -1e-3},
    )
    with pytest.raises(ValueError, match="net gain"):
        ece.solve_kinetic_ece_profile(
            90e9, [1.0], wavevector_rad_m=[[1, 0, 0]],
            magnetic_field_T=[[0, 0, 2]], polarization=[[1, 0, 0]],
            electron_density_m3=[1e19], electron_temperature_eV=[100],
        )
