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


def test_registration_rejects_parameter_names_that_duplicate_after_trimming():
    with pytest.raises(ValueError, match="unique after trimming"):
        register_spectral_model(
            "duplicate_trimmed_names", lambda x, a, b: a * x + b,
            parameter_names=("amplitude", " amplitude "),
            initial_values=(1.0, 0.0),
        )


def test_fit_rejects_nonpositive_signal_uncertainty(registered_model):
    x = np.linspace(531.0, 533.0, 8)
    with pytest.raises(ValueError, match="finite and positive"):
        fit_custom_spectral_model(
            x, np.ones_like(x), model_name=registered_model,
            signal_uncertainty=np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        )


def test_fit_rejects_custom_model_with_wrong_output_shape():
    register_spectral_model(
        "wrong_shape_model", lambda x, a: np.asarray([a]),
        parameter_names=("a",), initial_values=(1.0,),
    )
    try:
        with pytest.raises(ValueError, match="matching wavelength shape"):
            fit_custom_spectral_model(
                np.linspace(1.0, 2.0, 5), np.ones(5), model_name="wrong_shape_model"
            )
    finally:
        unregister_spectral_model("wrong_shape_model")


def test_fit_rejects_nonfinite_custom_model_output():
    register_spectral_model(
        "nonfinite_model", lambda x, a: np.full(x.shape, np.nan),
        parameter_names=("a",), initial_values=(1.0,),
    )
    try:
        with pytest.raises(ValueError, match="finite values matching wavelength shape"):
            fit_custom_spectral_model(
                np.linspace(1.0, 2.0, 5), np.ones(5), model_name="nonfinite_model"
            )
    finally:
        unregister_spectral_model("nonfinite_model")


def test_registration_rejects_string_instead_of_parameter_name_sequence():
    with pytest.raises(ValueError, match="not a string"):
        register_spectral_model(
            "string_names", lambda x, a: a * x,
            parameter_names="amplitude", initial_values=(1.0,),
        )


def test_registration_rejects_malformed_numeric_bounds():
    with pytest.raises(ValueError, match="numeric lower and upper sequences"):
        register_spectral_model(
            "malformed_bounds", lambda x, a: a * x,
            parameter_names=("a",), initial_values=(1.0,),
            bounds=("not-a-bound-sequence", (2.0,)),
        )


def test_fit_rejects_wrong_uncertainty_shape(registered_model):
    with pytest.raises(ValueError, match="match the signal shape"):
        fit_custom_spectral_model(
            np.linspace(531.0, 533.0, 10), np.ones(10),
            model_name=registered_model, signal_uncertainty=np.ones(9),
        )


def test_fit_rejects_non_boolean_absolute_sigma(registered_model):
    with pytest.raises(ValueError, match="absolute_sigma must be boolean"):
        fit_custom_spectral_model(
            np.linspace(531.0, 533.0, 10), np.ones(10),
            model_name=registered_model, absolute_sigma="yes",
        )


def test_fit_requires_more_samples_than_parameters(registered_model):
    with pytest.raises(ValueError, match="more spectral samples"):
        fit_custom_spectral_model(
            np.linspace(531.0, 533.0, 4), np.ones(4), model_name=registered_model,
        )


def test_unregister_returns_false_for_missing_model():
    assert unregister_spectral_model("model_that_does_not_exist") is False


def test_model_name_whitespace_is_normalized_consistently(registered_model):
    x = np.linspace(531.0, 533.0, 31)
    y = 4.0 * np.exp(-0.5 * ((x - 532.0) / 0.18) ** 2) + 0.5
    out = fit_custom_spectral_model(x, y, model_name=f" {registered_model} ")
    assert out["model_name"] == registered_model
    assert unregister_spectral_model(f" {registered_model} ") is True
    assert unregister_spectral_model(registered_model) is False


def test_unregister_invalid_or_blank_name_is_safe():
    assert unregister_spectral_model("") is False
    assert unregister_spectral_model("   ") is False
    assert unregister_spectral_model(None) is False


def test_custom_model_physical_mapper_requires_complete_explicit_contract():
    def model(x, amplitude, center, sigma, offset):
        return amplitude*np.exp(-0.5*((x-center)/sigma)**2)+offset

    with pytest.raises(ValueError, match="physical_contract mapping is required"):
        register_spectral_model(
            "missing_physical_contract", model,
            parameter_names=("amplitude","center","sigma","offset"),
            initial_values=(8.0,532.0,.2,1.0),
            physical_parameter_mapper=lambda p: {"electron_temperature_eV": 100.0},
        )


def test_custom_model_physical_mapper_emits_provenance_and_units():
    def model(x, amplitude, center, sigma, offset):
        return amplitude*np.exp(-0.5*((x-center)/sigma)**2)+offset
    name="contracted_gaussian"
    register_spectral_model(
        name, model,
        parameter_names=("amplitude","center","sigma","offset"),
        initial_values=(8.0,532.0,.2,1.0),
        physical_parameter_mapper=lambda p: {"electron_temperature_eV": float(p["sigma"]**2*1e4)},
        physical_contract={
            "output_units":{"electron_temperature_eV":"eV"},
            "formula_references":["user-supplied model relation: T_e = 10^4 sigma^2"],
            "assumptions":["synthetic test-only calibration"],
            "limitations":["not a Thomson-scattering validated relation"],
            "parameter_provenance":{"electron_temperature_eV":"derived from fitted sigma using explicit mapper"},
        },
    )
    try:
        x=np.linspace(531.0,533.0,101)
        y=10*np.exp(-.5*((x-532.1)/.16)**2)+2
        out=fit_custom_spectral_model(x,y,model_name=name)
        assert out["physical_inference"]["values"]["electron_temperature_eV"] == pytest.approx(256.0,rel=1e-5)
        assert out["physical_inference"]["contract"]["output_units"]["electron_temperature_eV"]=="eV"
        assert out["physical_inference"]["status"]=="model_contract_mapped_with_local_covariance_not_independently_validated"
        assert out["physical_inference"]["uncertainties"]["electron_temperature_eV"] >= 0
    finally:
        unregister_spectral_model(name)
