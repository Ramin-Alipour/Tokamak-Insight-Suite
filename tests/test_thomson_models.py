import numpy as np
import pytest

from tokamak_analysis.thomson_models import (
    fit_custom_spectral_model,
    list_spectral_models,
    register_spectral_model,
    unregister_spectral_model,
)


@pytest.fixture
def registered_model():
    name = "test_gaussian_linear"

    def model(x, amplitude, center, sigma, offset):
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset

    register_spectral_model(
        name, model,
        parameter_names=("amplitude", "center", "sigma", "offset"),
        initial_values=(8.0, 532.0, 0.2, 1.0),
        bounds=((0.0, 530.0, 0.001, -10.0), (20.0, 534.0, 2.0, 10.0)),
        description="Test-only Gaussian spectrum",
    )
    yield name
    unregister_spectral_model(name)


def test_custom_model_registry_and_fit_recovers_parameters(registered_model):
    x = np.linspace(531.0, 533.0, 101)
    y = 10.0 * np.exp(-0.5 * ((x - 532.1) / 0.16) ** 2) + 2.0
    out = fit_custom_spectral_model(
        x, y, model_name=registered_model, signal_uncertainty=np.full(x.size, 0.1)
    )
    assert out["parameters"]["amplitude"] == pytest.approx(10.0, rel=1e-5)
    assert out["parameters"]["center"] == pytest.approx(532.1, abs=1e-5)
    assert out["parameters"]["sigma"] == pytest.approx(0.16, rel=1e-5)
    assert out["parameters"]["offset"] == pytest.approx(2.0, abs=1e-5)
    assert out["fitted_spectrum"].shape == x.shape
    assert out["residual"].shape == x.shape
    assert out["reduced_chi_square"] is not None
    assert any(item["name"] == registered_model for item in list_spectral_models())


def test_registration_rejects_duplicate_names_without_replace(registered_model):
    with pytest.raises(ValueError, match="already registered"):
        register_spectral_model(
            registered_model, lambda x, a: a * x,
            parameter_names=("a",), initial_values=(1.0,),
        )


def test_fit_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown custom spectral model"):
        fit_custom_spectral_model(np.arange(5.0), np.arange(5.0), model_name="missing")


def test_fit_rejects_nonmonotonic_axis(registered_model):
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_custom_spectral_model(
            [531.0, 532.0, 531.5, 533.0, 534.0],
            [1, 2, 3, 4, 5], model_name=registered_model,
        )


def test_registration_validates_bounds():
    with pytest.raises(ValueError, match="Initial values must lie within bounds"):
        register_spectral_model(
            "bad_bounds", lambda x, a: a * x,
            parameter_names=("a",), initial_values=(5.0,),
            bounds=((0.0,), (2.0,)),
        )
