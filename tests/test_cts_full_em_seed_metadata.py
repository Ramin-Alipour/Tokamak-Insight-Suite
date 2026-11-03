import numpy as np
from types import SimpleNamespace


def test_full_em_cts_reports_which_optimizer_seeds_were_automatic(monkeypatch):
    import scipy.optimize
    import tokamak_analysis.cts_full_em as full_em
    import tokamak_analysis.cts_analysis as electrostatic

    frequency = np.linspace(-4.0e9, 4.0e9,  nine := 9)
    measured = np.linspace(1.0, 2.0, nine)
    species = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    monkeypatch.setattr(
        electrostatic,
        "estimate_cts_optimizer_initial_parameters",
        lambda *args, **kwargs: {
            "electron_density_m3": 2.0e19,
            "electron_temperature_eV": 1200.0,
            "ion_temperature_eV": 800.0,
            "method": "test_seed",
            "density_seed_assumption": "test alpha=1 seed only",
        },
    )

    def fake_forward(shift, *args, **kwargs):
        return {
            "spectral_density": np.linspace(0.2, 1.0, len(shift)),
            "fluctuation_covariance_density_current_E_B": np.zeros((2, 2)),
            "dielectric_tensor": np.eye(3, dtype=complex),
            "wave_tensor": np.eye(3, dtype=complex),
        }

    monkeypatch.setattr(full_em, "full_em_cts_spectral_density", fake_forward)

    def fake_least_squares(function, initial, **kwargs):
        return SimpleNamespace(
            x=np.asarray(initial, dtype=float),
            jac=np.eye(len(initial)),
            success=True,
            message="mocked deterministic fit",
            fun=np.zeros(len(initial), dtype=float),
            nfev=1,
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = full_em.analyze_full_em_cts_spectrum(
        frequency,
        measured,
        probe_frequency_Hz=140.0e9,
        scattering_geometry={
            "incident_direction": [1.0, 0.0, 0.0],
            "scattered_direction": [0.0, 1.0, 0.0],
        },
        ion_species=species,
        ion_fractions=[1.0],
        magnetic_field_T=[0.0, 0.0, 1.0],
        incident_polarization=[0.0, 1.0, 0.0],
        scattered_polarization=[0.0, 0.0, 1.0],
        instrument_response=np.eye(nine),
        numerical_convergence=False,
    )

    assert result["optimizer_initial_estimates"]["automatically_estimated_parameters"] == [
        "electron_density_m3",
        "electron_temperature_eV",
        "ion_temperature_eV",
    ]
    assert result["optimizer_initial_estimates"]["role"] == "optimizer_initial_estimates_only"
    assert result["optimizer"]["method"] == "deterministic_multistart_bounded_least_squares"
    assert result["optimizer"]["attempted_starts"] == 7
    assert result["optimizer"]["successful_starts"] == 7
