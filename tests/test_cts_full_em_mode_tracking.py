import numpy as np

from tokamak_analysis.cts_full_em import _select_flux_normalized_cold_mode


def test_cold_plasma_mode_selection_tracks_rotating_eigenpolarization(monkeypatch):
    from tokamak_analysis import cts_em_geometry

    def fake_modes(omega, direction, magnetic_field, species, *, mode_index):
        # A single physical branch rotates continuously from 0 to 80 degrees.
        # Its orthogonal companion remains separated by 90 degrees.
        angle = np.deg2rad(16.0 * (omega - 1.0))
        branch_angle = angle + mode_index * np.pi / 2.0
        field = np.array(
            [np.cos(branch_angle), np.sin(branch_angle), 0.0],
            dtype=complex,
        )
        return {
            "wavevector_rad_m": np.array([omega, 0.0, 0.0]),
            "electric_polarization_unit_flux": field,
            "refractive_index": 1.0,
            "poynting_vector_unit_flux": np.array([1.0, 0.0, 0.0]),
        }

    monkeypatch.setattr(
        cts_em_geometry, "solve_cold_plasma_propagating_modes", fake_modes
    )
    direction = np.array([1.0, 0.0, 0.0])
    magnetic_field = np.array([0.0, 0.0, 1.0])
    species = [{"density_m3": 1e18, "charge_C": -1.602176634e-19,
                "mass_kg": 9.1093837015e-31}]

    previous = np.array([1.0, 0.0, 0.0], dtype=complex)
    selected_branches = []
    for omega in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        _, field, diagnostics = _select_flux_normalized_cold_mode(
            omega, direction, magnetic_field, species, previous
        )
        selected_branches.append(diagnostics["mode_index"])
        previous = field

    # The same fixed launch polarization would select branch 1 at the final
    # frequency; overlap continuation correctly remains on branch 0.
    _, _, fixed_seed_diagnostics = _select_flux_normalized_cold_mode(
        6.0, direction, magnetic_field, species,
        np.array([1.0, 0.0, 0.0], dtype=complex),
    )
    assert selected_branches == [0, 0, 0, 0, 0, 0]
    assert fixed_seed_diagnostics["mode_index"] == 1
