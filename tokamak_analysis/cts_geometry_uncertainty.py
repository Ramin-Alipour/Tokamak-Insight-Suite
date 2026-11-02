"""Full-forward Monte-Carlo propagation of Full-EM CTS ray geometry uncertainty.

The geometry sampler must regenerate the equilibrium, trace probe and receiver
rays, propagate/re-supply local beam covariance, and recompute scattering-volume
overlap for every realization. Final-parameter covariance addition is not used
as a substitute for rerunning the geometry-dependent forward model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import numpy as np


def _validate_realization(value, index):
    if not isinstance(value, Mapping):
        raise ValueError(f"geometry sampler result {index} must be a mapping")
    geometry = value.get("scattering_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"geometry sampler result {index} needs scattering_geometry")
    records = geometry.get("frequency_resolved_geometry")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("a full frequency_resolved_geometry realization is required")
    for frequency_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"frequency geometry {frequency_index} must be a mapping")
        samples = record.get("scattering_samples")
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError(f"frequency geometry {frequency_index} has no scattering_samples")
        for sample_index, sample in enumerate(samples):
            required = (
                "incident_direction", "scattered_direction", "magnetic_field_T",
                "scattering_location_m", "sample_weight", "beam_overlap_m_minus3",
                "path_length_weight_m",
            )
            missing = [key for key in required if key not in sample]
            if missing:
                raise ValueError(
                    f"frequency geometry {frequency_index}, sample {sample_index} "
                    f"is missing {missing}"
                )
            weight = float(sample["sample_weight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("scattering sample weights must be finite and non-negative")
    return value


def _parameter_vector(result, names):
    species_t = result.get("ion_temperature_eV_by_species", {})
    species_u = result.get("ion_flow_projection_m_s_by_species", {})
    values = {
        "electron_density_m3": result.get("electron_density_m3"),
        "electron_temperature_eV": result.get("electron_temperature_eV"),
        "electron_flow_projection_m_s": result.get("electron_flow_projection_m_s"),
    }
    values.update({f"ion_temperature_eV:{key}": value for key, value in species_t.items()})
    values.update({f"ion_flow_projection_m_s:{key}": value for key, value in species_u.items()})
    try:
        vector = np.asarray([values[name] for name in names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Could not map the fitted CTS parameter names to physical outputs") from exc
    if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
        raise ValueError("fitted CTS parameter vector is unavailable or non-finite")
    return vector


def propagate_full_em_cts_ray_geometry_uncertainty(
    *,
    inference_kwargs: Mapping,
    geometry_sampler: Callable,
    samples=100,
    seed=84623,
    minimum_success_fraction=0.70,
):
    """Refit Full-EM CTS for complete, independently regenerated geometry draws.

    geometry_sampler(rng, index) must sample the equilibrium and beam-covariance
    inputs, call the ray tracer again, and return a mapping containing the full
    frequency_resolved_geometry. It may also return magnetic_field_T and the
    incident/scattered polarization vectors for that same realization.

    The routine performs one baseline fit and one complete spectral refit per
    accepted geometry draw. It reports the sample covariance of the fitted
    physical parameters, including cross-parameter terms. This covariance is
    conditional on the sampler's uncertainty model; correlations with other
    external uncertainty sources are not inferred.
    """
    from .cts_full_em import analyze_full_em_cts_spectrum

    if not isinstance(inference_kwargs, Mapping):
        raise ValueError("inference_kwargs must be a mapping")
    if not callable(geometry_sampler):
        raise ValueError("geometry_sampler must be callable")
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 20:
        raise ValueError("samples must be an integer >= 20")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    threshold = float(minimum_success_fraction)
    if not np.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
        raise ValueError("minimum_success_fraction must lie in [0.5, 1]")
    if "scattering_geometry" not in inference_kwargs:
        raise ValueError("inference_kwargs must include baseline scattering_geometry")
    if inference_kwargs.get("external_input_covariance") is not None:
        raise ValueError(
            "Use one joint sampler for ray geometry and local external inputs; "
            "separate covariance sources would lose their cross-correlation."
        )

    baseline_kwargs = copy.deepcopy(dict(inference_kwargs))
    baseline = analyze_full_em_cts_spectrum(**baseline_kwargs)
    names = list(baseline.get("parameter_names", []))
    if not names or len(set(names)) != len(names):
        raise ValueError("baseline fit did not return a unique parameter_names ordering")
    _parameter_vector(baseline, names)  # Validate that every named physical parameter is available.
    conditional_covariance = np.asarray(
        baseline.get("conditional_fit_covariance", baseline.get("fit_covariance")),
        dtype=float,
    )
    if conditional_covariance.shape != (len(names), len(names)) or not np.all(np.isfinite(conditional_covariance)):
        raise ValueError("baseline conditional fit covariance does not match parameter_names")

    rng = np.random.default_rng(int(seed))
    draws, failures = [], []
    for index in range(int(samples)):
        try:
            realization = _validate_realization(geometry_sampler(rng, index), index)
            kwargs = copy.deepcopy(baseline_kwargs)
            kwargs["scattering_geometry"] = realization["scattering_geometry"]
            for key in ("magnetic_field_T", "incident_polarization", "scattered_polarization"):
                if key in realization:
                    kwargs[key] = realization[key]
            kwargs["external_input_covariance"] = None
            kwargs["external_input_parameter_order"] = None
            kwargs["instrument_response_uncertainty"] = None
            result = analyze_full_em_cts_spectrum(**kwargs)
            if not result.get("optimizer_success", False):
                raise RuntimeError("Full-EM optimizer did not converge for this geometry draw")
            if list(result.get("parameter_names", [])) != names:
                raise ValueError("parameter ordering changed between geometry realizations")
            draws.append(_parameter_vector(result, names))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, KeyError) as exc:
            failures.append({"realization": int(index), "reason": str(exc)})

    required = max(12, int(np.ceil(threshold * int(samples))))
    output = dict(baseline)
    diagnostics = {
        "method": "full_forward_monte_carlo_geometry_refit",
        "requested_samples": int(samples),
        "successful_samples": len(draws),
        "failed_samples": len(failures),
        "minimum_successful_samples": required,
        "seed": int(seed),
        "parameter_order": names,
        "sampler_contract": (
            "Each realization resamples equilibrium and local beam covariance, "
            "reruns incident/receiver ray tracing, and recomputes scattering locations, "
            "path-length quadrature and Gaussian overlap before refitting."
        ),
        "failed_realizations": failures,
        "propagated": False,
    }
    output["ray_geometry_uncertainty"] = diagnostics
    if len(draws) < required:
        diagnostics["warning"] = "Insufficient successful geometry refits; no geometry covariance was reported."
        return output

    draw_array = np.asarray(draws, dtype=float)
    geometry_covariance = np.cov(draw_array, rowvar=False, ddof=1)
    if geometry_covariance.shape != conditional_covariance.shape or not np.all(np.isfinite(geometry_covariance)):
        diagnostics["warning"] = "Geometry sample covariance is invalid; no covariance was reported."
        return output
    response_covariance_value = baseline.get("instrument_response_parameter_covariance")
    response_covariance = (
        np.zeros_like(conditional_covariance)
        if response_covariance_value is None
        else np.asarray(response_covariance_value, dtype=float)
    )
    if response_covariance.shape != conditional_covariance.shape or not np.all(np.isfinite(response_covariance)):
        raise ValueError("baseline instrument-response covariance does not match the fitted parameter vector")
    total_covariance = conditional_covariance + response_covariance + geometry_covariance
    output["ray_geometry_parameter_covariance"] = geometry_covariance
    output.setdefault("external_uncertainty_covariance_components", {})["ray_geometry"] = geometry_covariance
    output["fit_covariance"] = total_covariance
    output["parameter_uncertainties"] = np.sqrt(np.maximum(np.diag(total_covariance), 0.0))
    uncertainty_by_name = dict(zip(names, output["parameter_uncertainties"]))
    output["electron_density_uncertainty_m3"] = uncertainty_by_name["electron_density_m3"]
    output["electron_temperature_uncertainty_eV"] = uncertainty_by_name["electron_temperature_eV"]
    output["electron_flow_projection_uncertainty_m_s"] = uncertainty_by_name["electron_flow_projection_m_s"]
    output["ion_temperature_uncertainty_eV_by_species"] = {
        name.split(":", 1)[1]: float(uncertainty_by_name[name])
        for name in names if name.startswith("ion_temperature_eV:")
    }
    output["ion_flow_projection_uncertainty_m_s_by_species"] = {
        name.split(":", 1)[1]: float(uncertainty_by_name[name])
        for name in names if name.startswith("ion_flow_projection_m_s:")
    }
    output["parameter_covariance_scope"] = (
        "conditional local-fit covariance plus any successfully propagated baseline instrument-response "
        "covariance and full-forward Monte-Carlo ray/equilibrium/beam-geometry covariance; "
        "cross-covariance between response and geometry sources is not included"
    )
    diagnostics.update({
        "propagated": True,
        "covariance": geometry_covariance,
        "total_covariance": total_covariance,
        "sample_parameter_mean": np.mean(draw_array, axis=0),
        "sample_parameter_standard_deviation": np.std(draw_array, axis=0, ddof=1),
        "interpretation": (
            "Sample covariance from complete geometry-dependent nonlinear refits; "
            "conditional on the sampler's declared equilibrium and beam-covariance model."
        ),
    })
    return output
