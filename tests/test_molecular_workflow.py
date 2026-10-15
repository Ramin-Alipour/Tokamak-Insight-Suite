import numpy as np
import pytest

from tis.molecular import analyze_molecular_boltzmann_plot


def test_molecular_boltzmann_plot_recovers_excitation_temperature():
    energy = np.array([0.05, 0.12, 0.25, 0.41, 0.58, 0.72])
    temperature = 0.35
    wavelength = np.array([500, 510, 520, 530, 540, 550.], dtype=float)
    probability = np.array([1.0, 1.5, 2.0, 1.2, 2.2, 1.7])
    weight = np.array([1, 3, 5, 3, 7, 5.], dtype=float)
    population = 1e8 * np.exp(-energy / temperature)
    intensity = population * probability * weight / wavelength
    result = analyze_molecular_boltzmann_plot(energy, intensity, wavelength, probability, weight)
    assert result["excitation_temperature_eV"] == pytest.approx(temperature, rel=1e-10)
    assert result["r_squared"] == pytest.approx(1.0)


def test_molecular_boltzmann_plot_rejects_nonpositive_intensity():
    with pytest.raises(ValueError, match="intensities"):
        analyze_molecular_boltzmann_plot(
            [0, 1, 2], [1, 0, 3], [500, 510, 520], [1, 1, 1], [1, 1, 1]
        )


def test_molecular_boltzmann_plot_requires_distinct_energies():
    with pytest.raises(ValueError, match="distinct upper-state energies"):
        analyze_molecular_boltzmann_plot(
            [1, 1, 1], [1, 2, 3], [500, 510, 520], [1, 1, 1], [1, 1, 1]
        )
