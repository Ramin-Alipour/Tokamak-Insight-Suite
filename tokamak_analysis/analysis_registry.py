"""Canonical ordered registry for the user-requested Tokamak Insight Suite analysis list.

Entries point to executable numerical functions. A mapping does not imply that every
diagnostic's raw-instrument calibration or physical inference is fully modeled.
"""
ANALYSES = [
    {"id":1,"name":"Thomson Scattering","module":"thomson_workflow","function":"analyze_thomson_profile"},
    {"id":2,"name":"Plasma Equilibrium Reconstruction","module":"equilibrium_reconstruction","function":"reconstruct_equilibrium"},
    {"id":3,"name":"Plasma Shape Reconstruction","module":"shape_reconstruction","function":"analyze_plasma_shape"},
    {"id":4,"name":"Plasma Current Analysis","module":"plasma_current","function":"analyze_plasma_current"},
    {"id":5,"name":"Magnetic Field Analysis","module":"magnetic_field","function":"analyze_magnetic_field"},
    {"id":6,"name":"Beta Analysis","module":"beta","function":"analyze_beta"},
    {"id":7,"name":"Safety Factor Analysis","module":"safety_factor","function":"estimate_safety_factor"},
    {"id":8,"name":"Magnetic Axis Analysis","module":"magnetic_axis","function":"locate_magnetic_axis"},
    {"id":9,"name":"X-Point Analysis","module":"x_point","function":"find_x_point_candidates"},
    {"id":10,"name":"Shafranov Shift Analysis","module":"shafranov_shift","function":"shafranov_shift"},
    {"id":11,"name":"Plasma Volume Analysis","module":"plasma_geometry","function":"geometry_from_boundary"},
    {"id":12,"name":"Plasma Cross-Section Analysis","module":"plasma_geometry","function":"geometry_from_boundary"},
    {"id":13,"name":"Average Plasma Density","module":"average_density","function":"analyze_average_density"},
    {"id":14,"name":"Electron Temperature Analysis","module":"electron_temperature","function":"analyze_electron_temperature"},
    {"id":15,"name":"Ion Temperature Analysis","module":"ion_temperature","function":"analyze_ion_temperature"},
    {"id":16,"name":"Langmuir Probe Analysis","module":"probe_analysis","function":"analyze_probe_voltage_current"},
    {"id":17,"name":"Ball-Pen Probe Analysis","module":"ball_pen_probe","function":"analyze_ball_pen_probe"},
    {"id":18,"name":"Gundestrup Probe Analysis","module":"gundestrup_probe","function":"analyze_gundestrup"},
    {"id":19,"name":"Multipin Probe Analysis","module":"multipin_probe","function":"analyze_multipin_probe"},
    {"id":20,"name":"Plasma and Floating Potential Analysis","module":"potential_analysis","function":"analyze_potentials"},
    {"id":21,"name":"Doppler Spectroscopy","module":"doppler_spectroscopy","function":"analyze_doppler_spectrum"},
    {"id":22,"name":"Doppler Ion Temperature","module":"doppler_temperature","function":"ion_temperature_from_doppler_fwhm"},
    {"id":23,"name":"Doppler Flow Velocity","module":"doppler_flow","function":"doppler_flow_velocity"},
    {"id":24,"name":"Impurity Concentration Analysis","module":"impurity_concentration","function":"estimate_impurity_fraction"},
    {"id":25,"name":"Spectral Line Broadening","module":"spectral_broadening","function":"decompose_gaussian_linewidth"},
    {"id":26,"name":"Spectral Line Identification","module":"spectral_line_identification","function":"identify_spectral_lines"},
    {"id":27,"name":"Spectral Line Ratio Analysis","module":"spectral_line_ratio","function":"analyze_line_ratio"},
    {"id":28,"name":"Interferometry Density Analysis","module":"interferometry","function":"interferometry_density"},
    {"id":29,"name":"Reflectometry Analysis","module":"reflectometry","function":"cutoff_density"},
    {"id":30,"name":"Density Wave Analysis","module":"density_waves","function":"analyze_density_waves"},
    {"id":31,"name":"Plasma Wave Frequency and Spectrum","module":"plasma_wave_spectrum","function":"analyze_wave_signal"},
    {"id":32,"name":"Hard X-Ray Analysis","module":"hard_xray","function":"analyze_hard_xray"},
    {"id":33,"name":"Soft X-Ray Analysis","module":"soft_xray","function":"analyze_soft_xray"},
    {"id":34,"name":"X-Ray Spectrum Analysis","module":"xray_spectrum","function":"analyze_xray_spectrum"},
    {"id":35,"name":"Bolometry Analysis","module":"bolometry","function":"analyze_bolometry"},
    {"id":36,"name":"Radiation Tomography","module":"radiation_tomography","function":"reconstruct_emissivity"},
    {"id":37,"name":"Visible Imaging Analysis","module":"visible_imaging","function":"analyze_visible_image"},
    {"id":38,"name":"Plasma Line Radiation Analysis","module":"line_radiation","function":"analyze_line_radiation"},
    {"id":39,"name":"Bolometric Tomography","module":"radiation_tomography","function":"reconstruct_emissivity"},
    {"id":40,"name":"Neutron Flux Analysis","module":"neutron_flux","function":"analyze_neutron_flux"},
    {"id":41,"name":"Neutron Energy Spectrum","module":"neutron_spectrum","function":"analyze_neutron_spectrum"},
    {"id":42,"name":"Fusion Reaction Rate Analysis","module":"fusion_rate","function":"fusion_reaction_rate"},
    {"id":43,"name":"Nuclear Gamma-Ray Analysis","module":"nuclear_gamma","function":"analyze_gamma_spectrum"},
    {"id":44,"name":"Runaway Electron Analysis","module":"runaway_electrons","function":"analyze_runaway_events"},
    {"id":45,"name":"Energetic Particle Energy Distribution","module":"energetic_particles","function":"analyze_energy_distribution"},
    {"id":46,"name":"Particle Loss Analysis","module":"energetic_particles","function":"particle_loss_rate"},
    {"id":47,"name":"Energetic Particle Flux","module":"energetic_particles","function":"energetic_particle_flux"},
    {"id":48,"name":"Runaway Electron Energy Analysis","module":"runaway_energy","function":"estimate_runaway_energy"},
    {"id":49,"name":"Vacuum Pressure Analysis","module":"vacuum_pressure","function":"analyze_vacuum_pressure"},
    {"id":50,"name":"Gas Puffing Analysis","module":"gas_puffing","function":"analyze_gas_puffing"},
    {"id":51,"name":"Fueling Rate Analysis","module":"fueling_rate","function":"analyze_fueling_rate"},
    {"id":52,"name":"Neutral Gas Pressure and Density","module":"neutral_gas","function":"neutral_density"},
    {"id":53,"name":"Wall Recycling Analysis","module":"wall_recycling","function":"recycling_coefficient"},
    {"id":54,"name":"Mirnov Coil Analysis","module":"mirnov_analysis","function":"analyze_mirnov_signal"},
    {"id":55,"name":"MHD Mode Analysis","module":"mhd_modes","function":"estimate_mode_number"},
    {"id":56,"name":"Magnetic Mode Frequency","module":"magnetic_mode_frequency","function":"magnetic_mode_frequency"},
    {"id":57,"name":"FFT Analysis","module":"fft_analysis","function":"analyze_fft"},
    {"id":58,"name":"SVD Analysis","module":"svd_analysis","function":"analyze_svd"},
    {"id":59,"name":"Wavelet Analysis","module":"wavelet_analysis","function":"analyze_wavelet"},
    {"id":60,"name":"Plasma Turbulence and Fluctuation Analysis","module":"turbulence","function":"analyze_turbulence"},
    {"id":61,"name":"Radial Transport Analysis","module":"transport","function":"fluctuation_transport"},
    {"id":62,"name":"Poloidal Transport Analysis","module":"transport","function":"fluctuation_transport"},
    {"id":63,"name":"Reynolds Stress Analysis","module":"transport","function":"reynolds_stress"},
    {"id":64,"name":"Energy Confinement Time","module":"confinement","function":"energy_confinement_time"},
    {"id":65,"name":"Heat Transport and Heat Flux","module":"heat_flux","function":"analyze_heat_flux"},
    {"id":66,"name":"Electron Heating Analysis","module":"heating_analysis","function":"analyze_heating"},
    {"id":67,"name":"Ion Heating Analysis","module":"heating_analysis","function":"analyze_heating"},
    {"id":68,"name":"ECRH Analysis","module":"ecrh_icrh","function":"ecrh_resonance_field"},
    {"id":69,"name":"ICRH Analysis","module":"heating_analysis","function":"analyze_heating"},
    {"id":70,"name":"RF-Plasma Interaction Analysis","module":"rf_interaction","function":"analyze_rf_interaction"},
    {"id":71,"name":"Divertor Heat Flux","module":"divertor_analysis","function":"analyze_divertor_flux"},
    {"id":72,"name":"Plasma-Facing Component Heat Load","module":"divertor_analysis","function":"analyze_divertor_flux"},
    {"id":73,"name":"Wall Erosion and Sputtering","module":"divertor_analysis","function":"sputtering_yield"},
    {"id":74,"name":"Divertor Particle Flux","module":"divertor_analysis","function":"analyze_divertor_flux"},
    {"id":75,"name":"Edge Potential and Electric Field","module":"edge_field","function":"analyze_edge_field"},
    {"id":76,"name":"L-H Transition Analysis","module":"lh_transition","function":"detect_lh_transition"},
    {"id":77,"name":"Signal Calibration","module":"signal_calibration","function":"calibrate_signal"},
    {"id":78,"name":"Signal Denoising and Filtering","module":"denoising","function":"denoise_signal"},
    {"id":79,"name":"Measurement Uncertainty Analysis","module":"measurement_uncertainty","function":"uncertainty_summary"},
    {"id":80,"name":"Channel Synchronization","module":"channel_synchronization","function":"synchronize_channels"},
]


def get_analysis(analysis_id: int) -> dict:
    """Return a copy of a registered analysis entry by ordered list ID."""
    for entry in ANALYSES:
        if entry["id"] == int(analysis_id):
            return dict(entry)
    raise KeyError(f"Unknown analysis ID: {analysis_id}")


def resolve_analysis(analysis_id: int):
    """Import and return the registered callable for an analysis ID."""
    from importlib import import_module
    entry = get_analysis(analysis_id)
    module = import_module(f".{entry['module']}", package=__package__)
    return getattr(module, entry["function"])
