"""Explicit identifiability contracts for Thomson Analysis 1 outputs.

These classifications describe what the selected measured-spectrum model can
identify. They do not imply experimental validation or add assumptions to the
forward model.
"""
from __future__ import annotations


def optical_thomson_identifiability(model: str, *, absolute_density_available: bool) -> dict:
    primary = [
        {
            "parameter": "electron_temperature_eV",
            "unit": "eV",
            "basis": "inferred from the measured calibrated spectral shape under the explicitly selected optical model",
        },
        {
            "parameter": "electron_velocity_projection_m_s",
            "unit": "m s^-1",
            "basis": "inferred only as the component projected along the scattering wavevector",
        },
    ]
    if model == "non_collective_kappa":
        primary.append({
            "parameter": "kappa_index",
            "unit": "dimensionless",
            "basis": "inferred within the selected finite-variance isotropic Kappa/Student-t projected-distribution family",
        })
    density_requirements = [
        "absolute spectral intensity explicitly declared in W/nm",
        "laser power at the scattering volume",
        "interaction length",
        "collection solid angle",
        "optical efficiency",
        "incident/scattered directions and incident polarization",
    ]
    return {
        "classification_version": "1.0",
        "primary": primary,
        "conditional": [{
            "parameter": "electron_density_m3",
            "unit": "m^-3",
            "status": "computed" if absolute_density_available else "not_computed_missing_absolute_intensity_or_optical_inputs",
            "required_additional_inputs": density_requirements,
            "interpretation": "A normalized spectral shape alone does not identify absolute electron density.",
        }],
        "non_identifiable": [
            {
                "parameter": "three_dimensional_electron_flow_velocity",
                "reason": "One scattering geometry identifies only the scattering-wavevector projection.",
            },
            {
                "parameter": "arbitrary_electron_velocity_distribution",
                "reason": "A Maxwellian, Kappa or isotropic Maxwell-Juttner parameterization is not an unrestricted VDF inversion.",
            },
            {
                "parameter": "ion_temperature_and_ion_flow",
                "reason": "The selected non-collective optical electron-feature model does not infer ion parameters.",
            },
        ],
        "model": model,
        "no_automatic_model_selection": True,
    }


def cts_electrostatic_identifiability(*, fast_ion_model: str | None = None) -> dict:
    primary = [
        {"parameter": "electron_density_m3", "unit": "m^-3", "basis": "inferred from the measured collective spectral response"},
        {"parameter": "electron_temperature_eV", "unit": "eV", "basis": "inferred from the measured collective spectral response"},
        {"parameter": "ion_temperature_eV", "unit": "eV", "basis": "one common fitted ion temperature shared by the supplied ion species"},
        {"parameter": "electron_velocity_projection_m_s", "unit": "m s^-1", "basis": "fitted component along the scattering wavevector"},
        {"parameter": "ion_velocity_projection_m_s", "unit": "m s^-1", "basis": "one common fitted ion-flow projection shared by the supplied ion species"},
    ]
    conditional = []
    if fast_ion_model == "maxwellian":
        conditional.extend([
            {"parameter": "fast_ion_fraction", "unit": "dimensionless", "status": "computed_under_two_maxwellian_population_model"},
            {"parameter": "fast_ion_temperature_eV", "unit": "eV", "status": "computed_under_two_maxwellian_population_model"},
        ])
    elif fast_ion_model == "infer_tabulated_1d":
        conditional.append({
            "parameter": "fast_ion_velocity_distribution_1d",
            "unit": "m s^-1; probability density",
            "status": "regularized_inversion",
            "required_additional_inputs": ["velocity grid", "regularization strength", "reference VDF shape"],
        })
    elif fast_ion_model == "tabulated_1d":
        conditional.append({
            "parameter": "fast_ion_fraction",
            "unit": "dimensionless",
            "status": "computed_for_externally_supplied_fixed_1d_VDF_shape",
        })
    return {
        "classification_version": "1.0",
        "primary": primary,
        "conditional": conditional,
        "non_identifiable": [
            {"parameter": "three_dimensional_electron_and_ion_flow_vectors", "reason": "A single CTS geometry identifies only projections along the scattering wavevector."},
            {"parameter": "species_resolved_ion_temperatures_and_flows", "reason": "The electrostatic model uses one common ion temperature and one common ion-flow projection for all supplied species."},
            {"parameter": "ion_composition", "reason": "Ion species and charge-density fractions are fixed external inputs, not inferred by this inversion."},
            {"parameter": "absolute_transmitter_receiver_power_coupling", "reason": "The electrostatic model profiles a spectral amplitude and does not invert absolute coupling."},
        ],
        "model": "unmagnetized_electrostatic_longitudinal_CTS",
        "external_inputs_are_not_measurements": ["ion_species", "ion_fractions_charge_density"],
    }


def cts_full_em_identifiability(
    species_names: list[str],
    *,
    external_input_covariance_supplied: bool = False,
    external_input_covariance_propagated: bool = False,
    instrument_response_uncertainty_supplied: bool = False,
    instrument_response_uncertainty_propagated: bool = False,
) -> dict:
    """Classify Full-EM CTS parameters and report actual uncertainty provenance.

    The fitted plasma parameters remain conditional on supplied composition,
    magnetic field, geometry and polarization. Their *uncertainties*, however,
    can include refit-based propagation of supplied external-input and measured
    response covariance. The status must reflect the actual inversion result,
    not a static claim that those sources are never propagated.
    """
    for name, value in (
        ("external_input_covariance_supplied", external_input_covariance_supplied),
        ("external_input_covariance_propagated", external_input_covariance_propagated),
        ("instrument_response_uncertainty_supplied", instrument_response_uncertainty_supplied),
        ("instrument_response_uncertainty_propagated", instrument_response_uncertainty_propagated),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    if external_input_covariance_propagated and not external_input_covariance_supplied:
        raise ValueError("external-input covariance cannot be propagated unless it was supplied")
    if instrument_response_uncertainty_propagated and not instrument_response_uncertainty_supplied:
        raise ValueError("response uncertainty cannot be propagated unless it was supplied")

    primary = [
        {"parameter": "electron_density_m3", "unit": "m^-3", "basis": "fitted from the measured Full-EM CTS spectrum"},
        {"parameter": "electron_temperature_eV", "unit": "eV", "basis": "fitted from the measured Full-EM CTS spectrum"},
        {"parameter": "electron_flow_projection_m_s", "unit": "m s^-1", "basis": "fitted projection along the local scattering wavevector"},
    ]
    for name in species_names:
        primary.extend([
            {"parameter": f"ion_temperature_eV:{name}", "unit": "eV", "basis": "species-resolved fitted ion feature"},
            {"parameter": f"ion_flow_projection_m_s:{name}", "unit": "m s^-1", "basis": "species-resolved fitted projection along the local scattering wavevector"},
        ])

    if external_input_covariance_propagated:
        external_status = "propagated_by_joint_external_input_perturbation_and_full_spectrum_refitting"
    elif external_input_covariance_supplied:
        external_status = "supplied_but_not_propagated_insufficient_successful_refits"
    else:
        external_status = "not_supplied_external_inputs_held_fixed"
    if instrument_response_uncertainty_propagated:
        response_status = "propagated_by_measured_response_perturbation_and_full_spectrum_refitting"
    elif instrument_response_uncertainty_supplied:
        response_status = "supplied_but_not_propagated_insufficient_successful_refits"
    else:
        response_status = "not_supplied_response_held_fixed"

    return {
        "classification_version": "1.1",
        "primary": primary,
        "conditional": [
            {"parameter": "fast_ion_population_parameters", "status": "only_where_explicit_fast_ion_population_model_is_supplied_and_identifiable"},
            {"parameter": "electron_non_maxwellian_distribution", "status": "forward_model_only_for_current_spectrum_inversion; drifted relativistic VDF fit is not implemented"},
            {"parameter": "external_input_uncertainty_contribution", "status": external_status, "covariance_supplied": external_input_covariance_supplied, "covariance_propagated": external_input_covariance_propagated},
            {"parameter": "instrument_response_uncertainty_contribution", "status": response_status, "uncertainty_supplied": instrument_response_uncertainty_supplied, "uncertainty_propagated": instrument_response_uncertainty_propagated},
        ],
        "non_identifiable": [
            {"parameter": "three_dimensional_electron_and_ion_flow_vectors", "reason": "A single geometry identifies only local scattering-wavevector projections."},
            {"parameter": "ion_species_and_charge_fractions", "reason": "Composition is supplied and held fixed."},
            {"parameter": "magnetic_field_and_mode_polarization", "reason": "These remain supplied external physical inputs; propagating their covariance does not infer their values from the CTS spectrum."},
            {"parameter": "absolute_transmitter_receiver_coupling", "reason": "The spectral amplitude is profiled as a nuisance parameter, not absolutely inverted."},
        ],
        "model": "magnetized_full_electromagnetic_CTS",
        "species_order": list(species_names),
        "uncertainty_scope": "local fit covariance conditional on nominal external inputs, augmented only by explicitly supplied and successfully propagated response/external-input covariance components",
    }


def spectral_distribution_identifiability() -> dict:
    return {
        "classification_version": "1.0",
        "primary": [
            {"parameter": "spectral_centroid_nm", "unit": "nm", "basis": "signed background-subtracted wavelength-space moment"},
            {"parameter": "observed_sigma_nm", "unit": "nm", "basis": "observed wavelength-space width; not deconvolved"},
            {"parameter": "spectral_skewness", "unit": "dimensionless", "basis": "third standardized wavelength-space moment"},
            {"parameter": "spectral_excess_kurtosis", "unit": "dimensionless", "basis": "fourth standardized wavelength-space moment"},
            {"parameter": "gaussian_shape_fit", "unit": "signal units and nm", "basis": "descriptive Gaussian shape fit convolved with measured response"},
        ],
        "conditional": [
            {"parameter": "moment_uncertainties", "status": "available_only_when_pointwise_uncertainty_or_full_spectral_covariance_is_supplied"},
            {"parameter": "physical_electron_temperature_and_flow", "status": "provided by parent canonical optical Thomson fit, not inferred by this descriptive child routine"},
        ],
        "non_identifiable": [
            {"parameter": "electron_velocity_distribution_function", "reason": "Wavelength moments and a Gaussian shape descriptor do not uniquely determine an arbitrary electron VDF."},
            {"parameter": "instrument_corrected_spectral_width", "reason": "This routine applies the measured response as a forward operator and does not deconvolve or correct measured data."},
            {"parameter": "absolute_electron_density", "reason": "This child routine does not perform absolute optical-power inversion."},
        ],
        "model": "descriptive_measured_spectral_distribution",
    }


def spatial_profile_identifiability() -> dict:
    return {
        "classification_version": "1.0",
        "primary": [
            {"parameter": "channel_resolved_electron_temperature", "unit": "eV", "basis": "inferred from each measured calibrated spectrum by the parent physical model"},
            {"parameter": "channel_resolved_scattering_vector_flow_projection", "unit": "m s^-1", "basis": "inferred from each measured calibrated spectrum where identifiable"},
        ],
        "conditional": [
            {"parameter": "channel_resolved_electron_density", "unit": "m^-3", "status": "computed_only_where_absolute_W_per_nm_intensity_and_complete_optical_inputs_identify_density"},
            {"parameter": "continuous_radial_profile", "status": "numerical_interpolation_or_smoothing_of_identifiable_channel_estimates_only", "required_additional_inputs": ["at least two distinct measured spatial coordinates", "selected profile representation"]},
            {"parameter": "profile_uncertainty_band", "status": "requires available channel covariance; independent-channel fallback is explicitly labelled"},
        ],
        "non_identifiable": [
            {"parameter": "values_outside_measured_spatial_interval", "reason": "No extrapolation is performed."},
            {"parameter": "flux_surface_or_equilibrium_mapping", "reason": "A supplied radius coordinate alone does not identify magnetic flux surfaces or equilibrium geometry."},
            {"parameter": "additional_physical_information_from_interpolation", "reason": "PCHIP/GCV do not add independent measurements or a new plasma-physics inference model."},
        ],
        "model": "measured_spectra_to_channel_physics_then_spatial_representation",
    }
