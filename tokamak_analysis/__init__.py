"""Computational analysis modules for tokamak diagnostics and plasma parameters.

Each module exposes explicit numerical functions. Instrument calibration,
diagnostic geometry and model-dependent assumptions must be supplied by users.
"""
__all__ = [
    "analysis_registry", "average_density", "ball_pen_probe", "beta", "bolometry", "channel_synchronization",
    "confinement", "denoising", "density_waves", "divertor_analysis", "doppler_flow",
    "doppler_spectroscopy", "doppler_temperature", "ecrh_icrh", "edge_field",
    "electron_temperature", "energetic_particles", "equilibrium_reconstruction",
    "fft_analysis", "fueling_rate", "fusion_rate", "gas_puffing", "gundestrup_probe",
    "hard_xray", "heat_flux", "heating_analysis", "impurity_concentration",
    "interferometry", "ion_temperature", "lh_transition", "line_radiation",
    "magnetic_axis", "magnetic_field", "magnetic_mode_frequency", "measurement_uncertainty",
    "mhd_modes", "mirnov_analysis", "multipin_probe", "neutral_gas", "neutron_flux",
    "neutron_spectrum", "nuclear_gamma", "plasma_current", "plasma_geometry",
    "plasma_wave_spectrum", "potential_analysis", "probe_analysis", "radiation_tomography",
    "reflectometry", "rf_interaction", "runaway_electrons", "runaway_energy",
    "safety_factor", "shafranov_shift", "shape_reconstruction", "signal_calibration",
    "soft_xray", "spectral_broadening", "spectral_line_identification", "spectral_line_ratio",
    "svd_analysis", "thomson_scattering", "thomson_workflow", "transport", "turbulence",
    "vacuum_pressure", "visible_imaging", "wall_recycling", "wavelet_analysis", "x_point",
    "xray_spectrum",
]
