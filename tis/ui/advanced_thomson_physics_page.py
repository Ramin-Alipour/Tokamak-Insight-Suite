"""Streamlit entry point for explicit advanced Analysis-1 physics solvers.

The numerical kernels live in tokamak_analysis.cts_full_physics. This page only
collects already-calibrated/user-prepared arrays and forwards them unchanged;
it performs no calibration, geometry reconstruction, or hidden model fallback.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
import numpy as np
from scipy.special import ndtr, stdtr

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def _example_wavelength_edges(wavelength_nm):
    wavelength = np.asarray(wavelength_nm, dtype=float)
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return edges


def _example_gaussian_bin_density(wavelength_nm, centre_nm, sigma_nm):
    edges = _example_wavelength_edges(wavelength_nm)
    return np.diff(ndtr((edges - centre_nm) / sigma_nm)) / np.diff(edges)


def _example_bimaxwellian_spectrum(wavelength_nm, laser_nm, scattering_angle_rad,
                                   temperature_eV, flow_m_s=0.0):
    q_factor = 2.0 * np.sin(scattering_angle_rad / 2.0)
    sigma_v = np.sqrt(1.602176634e-19 * temperature_eV / 9.1093837015e-31)
    centre = laser_nm * (1.0 - q_factor * flow_m_s / 299792458.0)
    sigma_lambda = laser_nm * q_factor * sigma_v / 299792458.0
    return _example_gaussian_bin_density(wavelength_nm, centre, sigma_lambda)


def _example_kappa_bin_density(wavelength_nm, laser_nm, q_factor, flow_m_s,
                               temperature_eV, kappa):
    edges = _example_wavelength_edges(wavelength_nm)
    e_charge, electron_mass, light_speed = 1.602176634e-19, 9.1093837015e-31, 299792458.0
    theta2 = ((2.0 * kappa - 3.0) / kappa) * e_charge * temperature_eV / electron_mass
    theta = np.sqrt(theta2)
    dof = 2.0 * kappa - 1.0
    scale = theta * np.sqrt(kappa / dof)
    velocity_edges = (light_speed / q_factor) * (1.0 - edges / laser_nm) - flow_m_s
    upper_standardized = velocity_edges[:-1] / scale
    lower_standardized = velocity_edges[1:] / scale
    probability = stdtr(dof, upper_standardized) - stdtr(dof, lower_standardized)
    positive_tail = lower_standardized > 0.0
    probability[positive_tail] = (
        stdtr(dof, -lower_standardized[positive_tail])
        - stdtr(dof, -upper_standardized[positive_tail])
    )
    return np.maximum(probability, 0.0) / np.diff(edges)


MODEL_LABELS = {
    "thomson_temperature_from_spectral_width": "Optical Thomson — conditional Maxwellian temperature from intrinsic thermal width with joint covariance",
    "thomson_kinetic_background_fit": "Optical Thomson — finite-bin Kappa / relativistic Maxwell-Juttner inference with structured background and joint covariance",
    "thomson_multi_population_relativistic_fit": "Optical Thomson — joint inference of 2–6 relativistic Maxwell-Juttner populations, simplex fractions and shared flow",
    "thomson_multiview_3d_evdf": "Optical Thomson — rank-gated 3-D EVDF inversion with vector flow, velocity-covariance tensor and propagated moment uncertainty from independent measured views",
    "kappa_electrostatic_susceptibility": "CTS — dimensional isotropic-Kappa susceptibility from the shared full-tail Landau kernel, with exact thermal scaling and analytic phase derivative",
    "kappa_dispersion_function": "Optical/CTS kinetic kernel — full-tail generalized Kappa plasma dispersion integral with exact Landau boundary value",
    "bi_kappa_electrostatic_susceptibility": "CTS — gyrotropic bi-Kappa directional Landau susceptibility with exact projected marginal",
    "thomson_projected_bi_kappa_evdf": "Optical Thomson — exact projected marginal of a gyrotropic 3-D bi-Kappa EVDF",
    "maxwellian_electrostatic_susceptibility": "CTS — analytic Fried-Conte Maxwellian Landau susceptibility (Faddeeva function)",
    "cts_collective_kappa_spectrum": "CTS — collective kinetic Maxwellian/Kappa/gyrotropic bi-Kappa electron and ion mixture dynamic structure spectrum",
    "cts_collective_kappa_inference": "CTS — infer electron density, temperature and Kappa index from measured collective spectrum",
    "cts_tabulated_evdf_susceptibility": "CTS — tabulated projected-EVDF susceptibility and analytic Landau phase derivative",
    "cts_tabulated_evdf_landau": "CTS — tabulated projected-EVDF Landau response and multispecies electrostatic spectrum (finite-eta or exact Plemelj limit)",
    "cts_multi_population_tabulated_evdf_landau": "CTS — density-weighted multi-electron-population EVDF Landau response and fluctuation spectrum",
    "cts_non_gyrotropic_gaussian_tensor_spectrum": "CTS — non-gyrotropic 3-D Gaussian temperature tensor projected exactly onto q and evaluated with the tabulated Landau response",
    "cts_multi_ion_flow_inference": "CTS — conditional inference of independent ion-species flow projections",
    "cts_electrostatic_measured_spectrum_inference": "CTS — measured-spectrum electrostatic inversion with thermal and tabulated fast-ion populations",
    "thomson_projected_evdf": "Optical Thomson — response-aware 1-D projected electron velocity distribution inversion",
    "thomson_relativistic_exact_flow_projection": "Optical Thomson — exact relativistic Doppler inversion of a single q-parallel flow projection",
    "thomson_relativistic_projected_evdf": "Optical Thomson — relativistic projected EVDF (Thomson or Compton/Klein–Nishina)",
    "thomson_psf_profile_inverse": "Thomson profile — PSF/mixing-aware inversion with optional GCV regularization",
    "thomson_joint_spatial_spectra": "Thomson profile — joint wavelength-resolved inversion of measured spatial spectra through a supplied mixing/PSF operator",
    "thomson_three_dimensional_flow": "Optical Thomson — 3-D electron flow from multiple scattering views",
    "thomson_full_temperature_tensor": "Optical Thomson — unconstrained-gyrotropy 3-D electron temperature tensor from multi-view spectra",
    "thomson_full_temperature_tensor_spectral_fit": "Optical Thomson — general 3-D non-gyrotropic temperature tensor fitted directly to measured multi-view spectra",
    "gyrotropic_bi_maxwellian_anisotropy": "Optical Thomson — gyrotropic electron temperature anisotropy from multiple pitch angles",
    "gyrotropic_bi_maxwellian_spectral_fit": "Optical Thomson — joint calibrated-spectrum fit for gyrotropic bi-Maxwellian anisotropy",
    "gyrotropic_bi_kappa_spectral_fit": "Optical Thomson — joint multi-view gyrotropic anisotropic Kappa inference with relativistic-tail validity diagnostic",
    "cts_conserving_bgk_fdt": "CTS — conserving multispecies BGK response + Poisson screening + classical matrix FDT",
    "landau_fdt_cts": "Multispecies Landau operator + matching FDT + Poisson-screened electrostatic CTS",
    "landau_fdt_full_em_constitutive": "Full-EM CTS — conserving Landau–FDT Cartesian current-response kernel (not a three-wave spectrum)",
    "polarized_kinetic_ece": "Magnetized kinetic ECE opacity + two-mode polarized transfer + stratified mode conversion",
    "stratified_nonparaxial_vector": "Non-paraxial vector angular spectrum + anisotropic Berreman propagation",
    "periodic_3d_full_wave": "CTS — 3-D Bloch-periodic full-vector Maxwell solve with Poynting/energy closure",
    "stratified_mode_conversion": "Coherent oblique-incidence Berreman/Jones mode conversion",
    "relativistic_kinetic_ece_profile": "ECE — segment-resolved relativistic gyrotropic kinetic absorption (supplied ray and modes)",
    "relativistic_kinetic_ece": "ECE — local relativistic gyrotropic Vlasov absorption kernel",
    "microwave_background": "CTS — free-free/ECE transfer, Planck/RJ brightness temperature and receiver-power closure",
    "multispecies_thermal_collision_matrix": "CTS — species-resolved thermal Coulomb collision matrix (not a kinetic fluctuation closure)",
    "quantum_fdt": "CTS — symmetrized quantum fluctuation-dissipation spectrum for a supplied causal response",
    "quantum_fdt_unsymmetrized": "CTS — unsymmetrized quantum S-greater spectrum with KMS detailed balance",
    "polarized_lte_stokes": "CTS — polarized Stokes LTE transfer with dichroism and Faraday rotation/conversion",
    "stratified_polarized_lte_stokes": "CTS — ordered segment-by-segment polarized LTE Stokes transfer",
    "cold_plasma_dielectric": "CTS — local multispecies cold-plasma dielectric tensor",
    "cold_plasma_magnetoionic_modes": "CTS — local cold-plasma magnetoionic modes and polarization eigenvectors",
    "cold_plasma_absorbed_power_density": "CTS — local collisional wave-absorption power density",
    "cold_plasma_mode_radiative_coefficients": "CTS — modal attenuation and explicitly selected LTE Kirchhoff emissivity",
    "cold_plasma_mode_group_velocity": "CTS — collisionless cold-plasma group velocity and wave-normal walk-off",
    "plasma_charge_state_moments": "CTS — quasineutrality, effective charge and multispecies charge moments",
    "thomson_normalized_poloidal_flux": "Optical Thomson — map channel positions to equilibrium-normalized poloidal flux rho_pol",
}

# Full-EM response and field-fluctuation routes are explicit Dashboard choices.
MODEL_LABELS["landau_fdt_full_em_spectrum"] = "Full-EM CTS — end-to-end Landau–FDT current noise → Maxwell field/Stokes spectrum"
MODEL_LABELS["landau_fdt_full_em_response"] = "Full-EM CTS — conserving Landau–FDT Cartesian current-response kernel"
MODEL_LABELS["landau_fdt_full_em_field_fluctuations"] = "Full-EM CTS — Maxwell propagation of Landau–FDT electric and magnetic fluctuations"

MODEL_PHYSICS_NOTES = {
    "cts_non_gyrotropic_gaussian_tensor_spectrum": (
        "A general symmetric positive-definite 3-D electron temperature tensor is marginalized exactly along "
        "q_hat=(n_scattered-n_incident)/|...|. The solver uses T_q=q_hat.T @ T_e @ q_hat and "
        "sigma_q^2=e*T_q/m_e, then evaluates the multispecies electrostatic Landau CTS response. "
        "The thermal marginal is zero-centred and q_hat dot electron_velocity_m_s supplies the Doppler shift once. "
        "One view does not identify all six tensor components; use independent scattering geometries for tensor inference. "
        "The kernel is non-relativistic and rejects insufficient velocity-grid tail coverage."
    ),
    "cts_collective_kappa_spectrum": (
        "Electron populations may use distribution='bi_kappa' with temperature_parallel_eV, "
        "temperature_perpendicular_eV, kappa and magnetic_field_unit. The electrostatic "
        "response uses the exact q-projected gyrotropic marginal T_q=T_perp+(T_parallel-T_perp)"
        "*(b_hat dot q_hat)^2. This is an unmagnetized longitudinal response to an anisotropic "
        "equilibrium; it is not a magnetized harmonic dielectric or Full-EM CTS vertex."
    ),
    "thomson_projected_bi_kappa_evdf": (
        "Exact marginal of a gyrotropic 3-D bi-Kappa distribution. For pitch cosine "
        "mu=b_hat dot q_hat, T_q=T_perp+(T_parallel-T_perp)*mu^2 and "
        "theta_q^2=((2*kappa-3)/kappa)*e*T_q/m_e. The returned 1-D PDF is normalized "
        "on the infinite domain; finite-grid probability coverage is reported without "
        "renormalizing missing tails. Requires kappa>3/2. One view identifies T_q only, "
        "not T_parallel and T_perpendicular independently."
    ),
    "landau_fdt_full_em_spectrum": (
        "End-to-end local Full-EM fluctuation calculation: assemble the conserving multispecies Landau operator and its matched FDT current-noise covariance, derive the constitutive response, then propagate it through the Maxwell Green dyadic. The default dielectric is the constitutive dielectric from the same solver; an explicit relative_dielectric_tensor may be supplied for a separately validated passive background. Scope is unmagnetized, zero-flow, common-temperature Maxwellian plasma; this is not yet a magnetized CTS three-wave vertex or calibrated receiver spectrum."
    ),
    "landau_fdt_full_em_field_fluctuations": (
        "Input constitutive_response is the mapping returned by landau_fdt_full_em_constitutive. "
        "The solver computes S_E = mu0^2 omega^2 G S_J G^dagger, S_B = (k_cross/omega) "
        "S_E (k_cross/omega)^dagger, and the spectral Poynting flux. Undamped Maxwell "
        "poles are rejected. This is not a three-wave CTS vertex. The solver also "
        "returns transverse electric coherency and I/Q/U/V Stokes spectral densities "
        "with V=2 Im(C12) for exp(-i omega t); the input current-noise covariance "
        "must be Hermitian positive semidefinite."
    ),
    "cts_tabulated_evdf_susceptibility": (
        "Tabulated projected 1-D EVDF susceptibility. landau_limit=true evaluates "
        "the exact first-order retarded boundary value using PV integral + i*pi*f'(v_phase), "
        "rather than an artificial finite pole displacement. It rejects phase velocities "
        "at derivative knots and does not expose the second-order pole derivative in this mode."
    ),
    "cts_multi_population_tabulated_evdf_landau": (
        "For electron populations j with density fractions alpha_j, the total longitudinal response is "
        "chi_e=sum_j alpha_j chi_j, where chi_j=-(omega_pe^2/k^2) times the unit-density integral "
        "f'_j(v)/(v-v_phase-u_j-i0) dv. The electron fluctuation source is the separate weighted sum "
        "sum_j alpha_j f_j(v_phase-u_j), not a single averaged distribution. The shared dielectric is "
        "epsilon=1+sum_j chi_j+sum_i chi_i; the ion source is screened by |chi_e|^2/|epsilon|^2. "
        "Fractions and rest-frame projected 1-D EVDFs are supplied explicitly; this is a non-relativistic, "
        "unmagnetized electrostatic forward model, not an inversion of component fractions or 3-D EVDFs."
    ),
    "cts_tabulated_evdf_landau": (
        "Electrostatic CTS spectrum from a supplied projected 1-D electron EVDF and either Maxwellian ions or independent tabulated 1-D ion marginals in each species rest frame. "
        "Set landau_limit=true to use the exact first-order Sokhotski-Plemelj response; otherwise "
        "pole_imaginary_part_m_s selects a finite retarded displacement. Exact-limit samples "
        "coincident with EVDF grid knots are rejected rather than regularized silently. "
        "For non-Maxwellian ions, supply ion_velocity_grids_m_s and ion_probability_densities_s_m "
        "as matching (n_ion_species, n_velocity) arrays in each species rest frame; "
        "ion_temperature_eV is optional for that branch. The output also reports projected means, second-moment directional temperatures, skewness and excess kurtosis."
    ),
    "thomson_temperature_from_spectral_width": (
        "Conditional Te estimator: Te = (sigma_lambda/lambda_0)^2 m_e c^2 / "
        "[4 sin^2(theta/2)]. Valid only for an intrinsic instrument-response-fitted "
        "isotropic non-relativistic Maxwellian thermal width; not a general temperature "
        "for Kappa, anisotropic, multi-population or flow-broadened spectra. Joint "
        "covariance order: [intrinsic width nm, laser wavelength nm, scattering angle rad]."
    ),
    "thomson_relativistic_exact_flow_projection": (
        "Exact mono-velocity q-parallel relativistic Doppler inversion. The optional "
        "input_covariance_nm2_rad2 is the full joint covariance of [fitted centre nm, "
        "laser wavelength nm, scattering angle rad] and propagates all cross terms "
        "through J C J^T. Supply either that matrix or the three legacy independent "
        "standard deviations, never both. This is not a 3-D flow or a broad-EVDF mean."
    ),
    "thomson_multiview_3d_evdf": (
        "Non-negative 3-D velocity-space inversion from independently measured scattering views. "
        "The stacked spectral kernel is assembled from the supplied view kernels; a data-only "
        "rank/conditioning gate determines whether the 3-D distribution is identifiable. "
        "Regularization can stabilize a conditional candidate but cannot create missing angular "
        "information. Supply calibrated kernels and uncertainty for every view."
    ),
    "bi_kappa_electrostatic_susceptibility": (
        "Gyrotropic 3-D bi-Kappa distribution projected exactly onto the supplied scattering vector, including electron bulk drift through v_phi,rest = v_phi,lab - u_e dot q_hat. "
        "The directional second-moment temperature is T_q = T_perp + (T_parallel-T_perp)*(b_hat dot q_hat)^2; "
        "the exact full-tail 1-D Kappa Landau integral and analytic phase derivative are then evaluated. "
        "One view identifies T_q only, not T_parallel and T_perpendicular separately. "
        "This is collisionless, non-relativistic, unmagnetized longitudinal susceptibility, not Full-EM CTS."
    ),
    "kappa_electrostatic_susceptibility": (
        "Collisionless, unmagnetized, non-relativistic longitudinal electron response for the exact "
        "1-D marginal of an isotropic 3-D finite-variance Kappa distribution (kappa > 3/2). "
        "The solver integrates the analytic Kappa derivative over the complete velocity axis in the electron rest frame, "
        "returns chi_e = -(omega_pe^2/k^2) integral f'_kappa(v)/(v-v_phase-i*eta) dv. "
        "It also returns d chi_e/d v_phase = -(omega_pe^2/k^2) integral "
        "f'_kappa(v)/(v-v_phase-i*eta)^2 dv at fixed positive eta, with separate real/imaginary "
        "quadrature error estimates, plus the fixed-k chain-rule derivative d chi_e/d omega = (1/k) d chi_e/d v_phase. The velocity grid is a tail-coverage gate, not a truncated-tail "
        "approximation. This remains an electron-only electrostatic response, not collisional, "
        "magnetized, relativistic, FDT or Full-EM CTS."
    ),
    "maxwellian_electrostatic_susceptibility": (
        "Exact collisionless unmagnetized longitudinal Maxwellian response in the electron rest frame, " 
        "with v_th=sqrt(2 e T_e/m_e), zeta=(v_phase_lab-u_e dot q_hat)/v_th and Z=i sqrt(pi) wofz(zeta). The optional signed electron_flow_projection_m_s shifts the Landau resonance; zero is the default. This is an electron susceptibility kernel only; it does not add ion response, FDT fluctuations, collisions, or Full-EM physics."
    ),
    "thomson_full_temperature_tensor_spectral_fit": (
        "Fits raw calibrated spectra jointly using a general symmetric positive-definite 3-D Gaussian velocity covariance. "
        "For each measured scattering direction q, T_q=q^T T_eV q, sigma_lambda=lambda_L*2*sin(theta/2)*sqrt(e*T_q/m_e)/c, "
        "and lambda_c=lambda_L*(1-2*sin(theta/2)*u_q/c). Gaussian probability is integrated exactly over finite wavelength bins; "
        "the measured response acts on bin-integrated power. Six independent view dyads and a full-rank spectral Jacobian are mandatory. "
        "This is a non-relativistic second-moment closure, not a general non-Gaussian 3-D EVDF."
    ),
    "thomson_full_temperature_tensor": (
        "Inverts T_q = q_hat.T @ T_eV @ q_hat for the six independent components of a general symmetric 3-D electron second-moment temperature tensor. " 
        "At least six independent scattering-view dyads are required; full covariance-weighted GLS and a positive-semidefinite tensor constraint are applied. " 
        "This removes the imposed gyrotropic-axis assumption but does not reconstruct higher EVDF moments. A PSD-boundary solution does not report symmetric local covariance."
    ),
    "gyrotropic_bi_maxwellian_anisotropy": (
        "Uses T_q = T_perp*(1-mu^2) + T_parallel*mu^2 with iterative "
        "errors-in-variables GLS. Optional magnetic_pitch_cosine_covariance "
        "and projected_temperature_pitch_cross_covariance are propagated "
        "together; their joint input covariance must be positive semidefinite. "
        "The physical inversion enforces strictly positive parallel and "
        "perpendicular temperatures. If a positivity bound is active, the "
        "reported local curvature covariance is not a symmetric confidence interval."
    ),
    "thomson_kinetic_background_fit": (
        "Fits calibrated measured optical Thomson spectra directly with the selected physical Kappa or relativistic Maxwell-Juttner forward model. "
        "The Kappa q-projected Student-t marginal is integrated exactly over finite wavelength bins; the relativistic kernel is Gauss-Legendre averaged inside each bin. "
        "The measured response acts on bin-integrated power. Physical amplitude and detector-background coefficients are fitted jointly, and the full local covariance retains physical/background cross terms. "
        "Supply measurement, model-discrepancy and systematic covariance separately. Kappa is isotropic, finite-variance (kappa > 3/2), and shape-only; MJ requires an explicit incident-polarization vector. "
        "No absolute density or unrestricted 3-D EVDF is inferred. A full-rank, well-conditioned data Jacobian is mandatory."
    ),
    "cts_collective_kappa_inference": (
        "Nonlinear measured-spectrum inversion of electron density, second-moment electron temperature "
        "and Kappa index. Ion composition, ion temperatures/flows, scattering geometry and measured "
        "instrument response are fixed explicit inputs. The data-only fit Jacobian must identify all "
        "three fitted parameters; otherwise the solver rejects the result. Optional electron_population_parameter_covariance "
        "supplies a joint covariance over every population's fraction, temperature ratio, Kappa index and projected flow; "
        "the solver propagates it through deterministic full-spectrum Monte Carlo refits and retains fit/input cross-covariance. "
        "Fractions must remain normalized in every accepted draw; invalid physical draws are rejected rather than clipped or renormalized. "
        "This is not a simultaneous ion-temperature or fast-ion inference and remains collisionless, unmagnetized electrostatic CTS."
    ),
    "cts_collective_kappa_spectrum": (
        "Collisionless, unmagnetized, longitudinal electrostatic CTS forward model. "
        "Electrons use exact projected 1-D marginals of isotropic Kappa, gyrotropic bi-Kappa, or Maxwellian populations with full Landau-contour susceptibility. "
        "A bi-Kappa population requires parallel/perpendicular second-moment temperatures and a unit magnetic-field direction; its directional temperature is T_q=T_perp+(T_parallel-T_perp)(b_hat dot q_hat)^2. "
        "Each population's density fraction, projected temperature and q-parallel flow are explicit; one scattering direction does not identify both anisotropy temperatures. "
        "Each ion species may independently use a Maxwellian, finite-variance Kappa or gyrotropic bi-Kappa projected marginal with its own projected flow; species charge neutrality and ion plasma-frequency weighting are retained. "
        "The longitudinal Landau-contour susceptibilities are summed before dielectric screening, and electron/ion fluctuation terms form S(k,omega). "
        "This remains collisionless, unmagnetized and electrostatic; it is not a collisional, relativistic or Full-EM closure."
    ),
    "cts_tabulated_evdf_susceptibility": (
        "Longitudinal susceptibility for a researcher-supplied projected 1-D electron VDF. "
        "The optional phase-velocity derivative uses the second-order Landau pole with "
        "the same positive retarded pole displacement; it is not a finite-difference estimate. "
        "The distribution is piecewise-linear with the declared zero extension and endpoint jumps."
    ),
    "cts_tabulated_evdf_landau": (
        "Forward model: evaluates the first-order Landau susceptibility from an explicitly "
        "supplied 1-D projected electron VDF using analytic interval integration of its "
        "piecewise-linear derivative and distributional endpoint jumps for the declared "
        "zero extension. Maxwellian multispecies ions and their flow projections are supplied "
        "separately. It does not infer the VDF, transverse velocity structure, or a general "
        "3-D EVDF; finite velocity support and pole displacement remain explicit assumptions."
    ),
    "thomson_relativistic_projected_evdf": (
        "Inverse model: exact relativistic Doppler kinematics in the Thomson limit, or "
        "single-electron Compton four-momentum conservation with polarized Klein-Nishina "
        "weighting when include_compton_recoil=true. The unknown is only the 1-D electron "
        "velocity marginal parallel to the scattering vector q; a single view does not "
        "identify the transverse velocity distribution or a general 3-D EVDF. The forward "
        "operator integrates over the velocity grid, applies the measured spectral response, "
        "and adds the supplied measured-space background. fitted_spectrum is the total "
        "background-plus-scattered prediction; fitted_scattered_spectrum excludes background. "
        "A regularized candidate is not promoted to a primary EVDF unless the data-only "
        "Jacobian passes the rank and conditioning gates."
    ),
    "gyrotropic_bi_maxwellian_spectral_fit": (
        "Physical forward model: exact Gaussian-CDF integration over each finite wavelength bin, "
        "followed by the measured response acting on bin-integrated spectral power. "
        "The result reports captured spectral probability without renormalizing truncated tails. "
        "A declared projected thermal-rms beta screen and a 0.1 bulk-flow beta screen guard the "
        "non-relativistic approximation. The public workflow also closes fitted per-view flow projections "
        "into a common 3-D vector when fit covariance and view geometry support it; optional geometry covariance "
        "and cross-covariance are propagated, with condition-number and common-flow residual reporting gates."
    ),
    "gyrotropic_bi_kappa_spectral_fit": (
        "Physical forward model: the projected bi-Kappa marginal is integrated exactly using "
        "its Student-t CDF (nu = 2*kappa - 1) over each finite wavelength bin, before measured "
        "response convolution. Wavelength-window tails are not renormalized. The public workflow also closes "
        "fitted per-view flow projections into a common 3-D vector when identifiable, with optional geometry "
        "covariance, condition-number and common-flow residual gates."
    ),
    "thomson_normalized_poloidal_flux": (
        "Maps externally supplied channel poloidal flux to rho_pol = sqrt((psi_pol-psi_axis)/(psi_sep-psi_axis)). "
        "Propagates the full joint covariance of channel fluxes, magnetic-axis flux and separatrix flux by the delta method. "
        "This does not reconstruct equilibrium or infer flux from R,Z; do not confuse rho_pol with rho_tor."
    ),
    "thomson_three_dimensional_flow": (
        "Infers a common electron bulk-flow vector from v_q,i = q_hat_i dot v_e using covariance-weighted GLS. "
        "The covariance-whitened design L^-1 Q must have rank three and condition number no greater than "
        "maximum_geometry_condition_number (default 1e3); otherwise the 3-D vector and covariance are withheld. "
        "For more than three views, the covariance-weighted projection residual is tested against the common-flow model; "
        "a chi-square survival probability below minimum_projection_consistency_p_value (default 1e-3) withholds the vector. "
        "Optional scattering-vector covariance and projection/geometry cross-covariance are propagated by "
        "iterative tangent-plane errors-in-variables GLS. This is a shared-flow multi-view model, not a local flow map."
    ),
}

MODEL_PHYSICS_NOTES["cts_conserving_bgk_fdt"] = (
    "Isothermal multispecies momentum-conserving BGK longitudinal response, "
    "self-consistent Poisson screening, and classical matrix FDT. Requires "
    "one species record per density/mass/charge and one collision frequency "
    "per species. Conserves each species particle number and total mixture "
    "momentum, but not total energy. Electrostatic only; not the magnetized "
    "Full-EM three-wave vertex. Zero frequency is rejected pending a "
    "separately converged static-limit extrapolation. The velocity integral "
    "is refined by doubled Gauss-Hermite orders and reports its relative "
    "response change; quantitative CTS release remains blocked until "
    "collision-model, frequency-grid and experimental-normalization checks pass."
)

EXAMPLES = {
    "thomson_multi_population_relativistic_fit": {
        "wavelength_nm": np.linspace(350.0, 750.0, 81).tolist(),
        "measured_spectrum": (np.exp(-0.5*((np.linspace(350.0, 750.0, 81)-532.0)/35.0)**2)).tolist(),
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": 1.5707963267948966,
        "instrument_response": np.eye(81).tolist(),
        "population_count": 2,
        "incident_polarization": [0.0, 0.0, 1.0],
        "flow_projection_initial_m_s": 0.0,
        "signal_uncertainty": np.full(81, 0.02).tolist(),
        "radial_order": 12,
        "angular_order": 8,
        "background_degree": 0
    },
    "cts_non_gyrotropic_gaussian_tensor_spectrum": {
        "wavelength_nm": np.linspace(1060.0, 1068.0, 41).tolist(),
        "laser_wavelength_nm": 1064.0,
        "scattering_angle_rad": 1.5707963267948966,
        "electron_density_m3": 1.0e18,
        "electron_temperature_tensor_eV": [[100.0, 20.0, 0.0], [20.0, 180.0, 10.0], [0.0, 10.0, 250.0]],
        "electron_velocity_m_s": [1.0e5, -0.5e5, 0.2e5],
        "electron_velocity_grid_m_s": np.linspace(-4.0e7, 4.0e7, 2401).tolist(),
        "ion_species": [{"name": "H+", "charge_state": 1.0, "mass_amu": 1.00784}],
        "ion_fractions": [1.0], "ion_temperature_eV": [10.0],
        "ion_velocity_m_s": [[0.0, 0.0, 0.0]],
        "incident_direction": [1.0, 0.0, 0.0], "scattered_direction": [0.0, 1.0, 0.0],
        "pole_imaginary_part_m_s": 100.0, "minimum_velocity_grid_probability": 0.999
    },
    "periodic_3d_full_wave": {
        "omega_rad_s": 31415926535.89793,
        "x_m": [0.0, 0.01, 0.02],
        "y_m": [0.0, 0.01, 0.02],
        "z_m": [0.0, 0.01, 0.02],
        "epsilon_r": {
            "real": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "imag": [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]
        },
        "impressed_current_density_A_m2": np.pad(
            np.ones((3, 3, 3, 1)), ((0, 0), (0, 0), (0, 0), (0, 2))
        ).tolist()
    },
    "kappa_dispersion_function": {
        "zeta_real": 0.7, "zeta_imag": 0.35, "kappa": 4.0
    },
    "thomson_projected_bi_kappa_evdf": {
        "velocity_grid_m_s": np.linspace(-1.2e8, 1.2e8, 2401).tolist(),
        "temperature_parallel_eV": 1500.0,
        "temperature_perpendicular_eV": 700.0,
        "kappa": 4.0,
        "magnetic_field_unit": [0.0, 0.0, 1.0],
        "scattering_vector_unit": [0.7071067811865476, 0.0, 0.7071067811865476],
        "flow_projection_m_s": 0.0
    },
    "landau_fdt_full_em_spectrum": {
        "angular_frequency_rad_s": [1.0e5, 2.0e5],
        "normalized_velocity_x": [-4.0, 0.0, 4.0],
        "normalized_velocity_y": [-4.0, 0.0, 4.0],
        "normalized_velocity_z": [-4.0, 0.0, 4.0],
        "species": [
            {"name": "electrons", "density_m3": 1.0e18, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "ions", "density_m3": 1.0e18, "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,
        "wave_number_m_inverse": 0.001,
        "propagation_direction": [1.0, 0.0, 0.0],
        "common_bulk_flow_m_s": [0.0, 0.0, 0.0],
        "coulomb_logarithm": 12.0,
        "condition_limit": 1.0e12
    },
    "landau_fdt_full_em_field_fluctuations": {
        "constitutive_response": {
            "angular_frequency_rad_s": [100000.0, 200000.0],
            "relative_dielectric_tensor": [
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
            ],
            "current_fluctuation_spectral_matrix": [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ]
        },
        "wave_number_m_inverse": 0.001,
        "propagation_direction": [1.0, 2.0, 0.5]
    },
    "cts_tabulated_evdf_susceptibility": {
        "velocity_grid_m_s": np.linspace(-8.0e6, 8.0e6, 1601).tolist(),
        "probability_density_s_m": np.exp(-0.5*(np.linspace(-8.0e6, 8.0e6, 1601)/1.0e6)**2).tolist(),
        "phase_velocity_m_s": 0.735e6, "electron_density_m3": 1.0e18,
        "wavenumber_m_inverse": 120.0, "landau_limit": True
    },
    "cts_multi_population_tabulated_evdf_landau": {
        "wavelength_nm": np.linspace(1060.0, 1068.0, 41).tolist(),
        "laser_wavelength_nm": 1064.0, "scattering_angle_rad": 1.5707963267948966,
        "electron_density_m3": 1.0e18,
        "electron_populations": [
            {
                "velocity_grid_m_s": np.linspace(-8.0e6, 8.0e6, 1601).tolist(),
                "probability_density_s_m": np.exp(-0.5*(np.linspace(-8.0e6, 8.0e6, 1601)/8.0e5)**2).tolist(),
                "density_fraction": 0.85, "bulk_velocity_m_s": [0.0, 0.0, 0.0]
            },
            {
                "velocity_grid_m_s": np.linspace(-8.0e6, 8.0e6, 1601).tolist(),
                "probability_density_s_m": np.exp(-0.5*(np.linspace(-8.0e6, 8.0e6, 1601)/2.0e6)**2).tolist(),
                "density_fraction": 0.15, "bulk_velocity_m_s": [0.0, 0.0, 0.0]
            }
        ],
        "ion_species": [{"name": "H+", "charge_state": 1.0, "mass_amu": 1.00784}],
        "ion_fractions": [1.0], "ion_temperature_eV": [10.0],
        "ion_velocity_m_s": [[0.0, 0.0, 0.0]],
        "incident_direction": [1.0, 0.0, 0.0], "scattered_direction": [0.0, 1.0, 0.0],
        "landau_limit": False, "pole_imaginary_part_m_s": 100.0
    },
    "cts_tabulated_evdf_landau": {
        "wavelength_nm": np.linspace(1060.0, 1068.0, 41).tolist(),
        "laser_wavelength_nm": 1064.0, "scattering_angle_rad": 1.5707963267948966,
        "electron_density_m3": 1.0e18,
        "electron_velocity_grid_m_s": np.linspace(-8.0e6, 8.0e6, 1601).tolist(),
        "electron_probability_density_s_m": np.exp(-0.5*(np.linspace(-8.0e6, 8.0e6, 1601)/1.0e6)**2).tolist(),
        "ion_species": [{"name": "H+", "charge_state": 1.0, "mass_amu": 1.00784}],
        "ion_fractions": [1.0], "ion_temperature_eV": [10.0],
        "electron_velocity_m_s": [0.0, 0.0, 0.0], "ion_velocity_m_s": [[0.0, 0.0, 0.0]],
        "incident_direction": [1.0, 0.0, 0.0], "scattered_direction": [0.0, 1.0, 0.0],
        "landau_limit": False, "pole_imaginary_part_m_s": 100.0
    },
    "thomson_temperature_from_spectral_width": {
        "intrinsic_sigma_nm": 0.02,
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": 1.5707963267948966,
        "sigma_intrinsic_sigma_nm": None,
        "sigma_laser_wavelength_nm": None,
        "sigma_scattering_angle_rad": None,
        "input_covariance_nm2_rad2": [
            [1.0e-8, 0.0, 0.0],
            [0.0, 4.0e-12, 0.0],
            [0.0, 0.0, 1.0e-8]
        ]
    },
    "bi_kappa_electrostatic_susceptibility": {
        "v": np.linspace(-2.0e8, 2.0e8, 1601).tolist(),
        "phase_velocity_m_s": 1.0e6,
        "electron_flow_projection_m_s": 2.0e5,
        "electron_density_m3": 2.0e19,
        "electron_temperature_parallel_eV": 1500.0,
        "electron_temperature_perpendicular_eV": 700.0,
        "magnetic_field_unit": [0.0, 0.0, 1.0],
        "scattering_vector_unit": [0.7071067811865476, 0.0, 0.7071067811865476],
        "kappa": 4.0,
        "wavenumber_m_inverse": 250.0,
        "minimum_velocity_grid_probability_coverage": 0.999
    },
    "maxwellian_electrostatic_susceptibility": {
        "phase_velocity_m_s": np.linspace(-2.0e6, 2.0e6, 81).tolist(),
        "electron_flow_projection_m_s": 1.5e5,
        "electron_density_m3": 2.0e19,
        "electron_temperature_eV": 1000.0,
        "wavenumber_m_inverse": 250.0
    },
    "thomson_normalized_poloidal_flux": {
        "poloidal_flux_wb": [0.11, 0.19, 0.34, 0.58],
        "magnetic_axis_flux_wb": 0.10,
        "separatrix_flux_wb": 0.60,
        "joint_flux_covariance": None
    },
    "thomson_kinetic_background_fit": {
        "wavelength_nm": np.linspace(529.0, 535.0, 181).tolist(),
        "measured_spectrum": (
            0.04 + 0.006*(np.linspace(529.0, 535.0, 181)-532.0)
            + 2.3*_example_kappa_bin_density(
                np.linspace(529.0, 535.0, 181), 532.0,
                2.0*np.sin(np.pi/4.0), 0.0, 300.0, 3.5
            )
        ).tolist(),
        "physical_model": "kappa",
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": 1.5707963267948966,
        "instrument_response": np.eye(181).tolist(),
        "physical_initial": [300.0, 0.0, 3.5],
        "background_model": "linear",
        "measurement_uncertainty": np.full(181, 0.003).tolist(),
        "model_discrepancy_covariance": None,
        "systematic_covariance": None,
        "include_compton_recoil": False,
        "maximum_recoil_correction_fraction": 0.01
    },
    "thomson_relativistic_exact_flow_projection": {
        "fitted_center_nm": 499.9998,
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "sigma_fitted_center_nm": None,
        "sigma_laser_wavelength_nm": None,
        "sigma_scattering_angle_rad": None,
        "input_covariance_nm2_rad2": [
            [1.0e-10, 0.0, 0.0],
            [0.0, 4.0e-12, 0.0],
            [0.0, 0.0, 1.0e-8]
        ]
    },
    "microwave_background": {
        "frequency_Hz": [90.0e9, 100.0e9, 110.0e9],
        "path_length_m": [0.05, 0.05],
        "electron_temperature_eV": [1500.0, 1500.0],
        "electron_density_m3": [1.0e19, 1.0e19],
        "ion_densities_m3": [[1.0e19], [1.0e19]],
        "ion_charge_states": [1],
        "gaunt_factor": [1.0, 1.0, 1.0],
        "incident_radiance_W_m2_sr_Hz": 0.0,
        "receiver_etendue_m2_sr": 1.0e-6,
        "end_to_end_efficiency": 0.7
    },
    "thomson_relativistic_projected_evdf": {
        "wavelength_nm": np.linspace(499.8, 500.2, 81).tolist(),
        "signal": (2.0 + 100.0*np.exp(-0.5*((np.linspace(499.8, 500.2, 81)-500.0)/0.07)**2)).tolist(),
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "instrument_response_matrix": np.eye(81).tolist(),
        "background_signal": np.full(81, 2.0).tolist(),
        "velocity_grid_m_s": np.linspace(-80000.0, 80000.0, 81).tolist(),
        "model_discrepancy_covariance": None,
        "include_compton_recoil": False,
        "maximum_recoil_correction_fraction": 0.01,
        "incident_polarization": [0.0, 0.0, 1.0],
        "regularization_strength": 0.001
    },
    "thomson_psf_profile_inverse": {
        "measured_channel_values": [2.0, 3.0, 4.0, 3.0, 2.0],
        "spatial_mixing_matrix": [[0.85,0.10,0.00,0.00,0.00],[0.10,0.80,0.10,0.00,0.00],[0.00,0.10,0.80,0.10,0.00],[0.00,0.00,0.10,0.80,0.10],[0.00,0.00,0.00,0.10,0.85]],
        "regularization_strength": None,
        "gcv_candidate_count": 31,
        "measurement_uncertainty": [0.1,0.1,0.1,0.1,0.1]
    },
    "thomson_multiview_3d_evdf": {
        "spectral_kernels": [[[1.0, 0.2], [0.5, 0.8], [0.1, 1.0]], [[0.2, 1.0], [0.8, 0.4], [1.0, 0.1]]],
        "measured_spectra": [[1.2, 1.3, 1.1], [1.2, 1.2, 1.1]],
        "velocity_grid_m_s": [[-1.0e5, 0.0, 0.0], [1.0e5, 0.0, 0.0]],
        "voxel_volume_m3_s3": [1.0, 1.0],
        "regularization_matrix": [[1.0, -1.0]],
        "measurement_uncertainty": [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
        "regularization_strength": 0.01
    },
    "thomson_projected_evdf": {
        "wavelength_nm": np.linspace(499.98, 500.02, 41).tolist(),
        "signal": (2.0 + 100.0*np.exp(-0.5*((np.linspace(499.98, 500.02, 41)-500.001)/0.003)**2)).tolist(),
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "instrument_response_matrix": np.eye(41).tolist(),
        "background_signal": np.full(41, 2.0).tolist(),
        "model_discrepancy_covariance": None,
        "systematic_covariance": None,
        "regularization_strength": 0.001,
        "velocity_limit_m_s": 8000.0,
        "velocity_points": 81
    },
    "cts_tabulated_evdf_susceptibility": {
        "velocity_grid_m_s": np.linspace(-8.0e6, 8.0e6, 1601).tolist(),
        "probability_density_s_m": np.exp(-0.5*(np.linspace(-8.0e6, 8.0e6, 1601)/1.0e6)**2).tolist(),
        "phase_velocity_m_s": 4.0e5,
        "electron_density_m3": 1.0e18,
        "wavenumber_m_inverse": 120.0,
        "pole_imaginary_part_m_s": 100.0,
        "include_phase_velocity_derivative": True
    },
    "cts_multi_ion_flow_inference": {
        "frequency_shift_Hz": np.linspace(-6.0e9, 6.0e9, 41).tolist(),
        "measured_spectrum": (2.0 + 80.0*np.exp(-0.5*(np.linspace(-6.0, 6.0, 41)/1.8)**2)).tolist(),
        "probe_frequency_Hz": 140.0e9,
        "scattering_geometry": {"scattering_angle_rad": 1.9198621771937625},
        "electron_density_m3": 2.0e19,
        "electron_temperature_eV": 1200.0,
        "ion_temperature_eV": [800.0, 900.0],
        "ion_species": [
            {"name": "D+", "charge_state": 1, "mass_amu": 2.014},
            {"name": "C6+", "charge_state": 6, "mass_amu": 12.0}
        ],
        "ion_fractions": [0.7, 0.3],
        "electron_velocity_projection_m_s": 0.0,
        "instrument_response": np.eye(41).tolist(),
        "ion_velocity_initial_m_s": {"D+": -40000.0, "C6+": 80000.0},
        "signal_uncertainty": np.full(41, 1.0).tolist(),
        "ion_velocity_bounds_m_s": [-500000.0, 500000.0]
    },
    "cts_electrostatic_measured_spectrum_inference": {
        "frequency_shift_Hz": np.linspace(-0.5e9, 0.5e9, 41).tolist(),
        "measured_spectrum": (2.0 + 80.0*np.exp(-0.5*(np.linspace(-0.5e9, 0.5e9, 41)/1.8e8)**2)).tolist(),
        "probe_frequency_Hz": 140.0e9,
        "scattering_geometry": {"scattering_angle_rad": 1.9198621771937625},
        "ion_species": [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        "ion_fractions": [1.0],
        "instrument_response": np.eye(41).tolist(),
        "signal_uncertainty": np.full(41, 1.0).tolist(),
        "electron_density_initial_m3": 2.0e19,
        "electron_temperature_initial_eV": 1200.0,
        "ion_temperature_initial_eV": 800.0,
        "electron_velocity_projection_m_s": 0.0,
        "ion_velocity_initial_m_s": 0.0,
        "inversion_method": "local_least_squares",
        "global_search": False,
        "fast_ion": {
            "species_index": 0,
            "kind": "tabulated_1d",
            "fraction_initial": 0.08,
            "fraction_bounds": [0.001, 0.5],
            "velocity_m_s": np.linspace(-2.0e6, 2.0e6, 81).tolist(),
            "probability_density_m_s_inverse": np.exp(-0.5*(np.linspace(-2.0e6, 2.0e6, 81)/3.0e5)**2).tolist(),
            "smoothness": 0.0
        }
    },
    "cts_collective_kappa_inference": {
        "wavelength_nm": np.linspace(499.7, 500.3, 81).tolist(),
        "signal": (0.02 + np.exp(-0.5*((np.linspace(499.7, 500.3, 81)-500.0)/0.08)**2)).tolist(),
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "response": np.eye(81).tolist(),
        "uncertainty": np.full(81, 0.01).tolist(),
        "incident_direction": [1.0, 0.0, 0.0],
        "scattered_direction": [0.0, 1.0, 0.0],
        "ion_species": [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        "ion_fractions": [1.0],
        "ion_temperature_eV": [200.0],
        "electron_velocity_m_s": [0.0, 0.0, 0.0],
        "ion_velocity_m_s": [[0.0, 0.0, 0.0]],
        "electron_populations": [
            {"density_fraction": 0.60, "temperature_ratio": 0.5, "kappa": 5.0, "distribution": "kappa", "flow_projection_m_s": 0.0},
            {"density_fraction": 0.25, "temperature_ratio": 1.5, "kappa": 3.5, "distribution": "kappa", "flow_projection_m_s": 0.0},
            {"density_fraction": 0.15, "temperature_ratio": 4.0, "distribution": "maxwellian", "flow_projection_m_s": 0.0}
        ],
        "initial_density_m3": 2.0e18,
        "initial_temperature_eV": 250.0,
        "initial_kappa": 4.0,
        "density_bounds_m3": [1.0e17, 1.0e20],
        "temperature_bounds_eV": [20.0, 2000.0],
        "kappa_bounds": [1.6, 30.0]
    },
    "cts_collective_kappa_spectrum": {
        "wavelength_nm": np.linspace(499.7, 500.3, 81).tolist(),
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "electron_density_m3": 2.0e18,
        "electron_temperature_eV": 250.0,
        "kappa": 4.0,
        "ion_species": [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        "ion_fractions": [1.0],
        "ion_temperature_eV": [200.0],
        "electron_velocity_m_s": [0.0, 0.0, 0.0],
        "ion_velocity_m_s": [[0.0, 0.0, 0.0]],
        "electron_populations": [
            {"density_fraction": 0.65, "temperature_parallel_eV": 420.0, "temperature_perpendicular_eV": 120.0, "magnetic_field_unit": [1.0, 0.0, 0.0], "kappa": 5.0, "distribution": "bi_kappa", "flow_projection_m_s": -100000.0},
            {"density_fraction": 0.25, "temperature_ratio": 2.5, "kappa": 3.0, "distribution": "kappa", "flow_projection_m_s": 150000.0},
            {"density_fraction": 0.10, "temperature_ratio": 5.0, "distribution": "maxwellian", "flow_projection_m_s": 800000.0}
        ],
        "ion_populations": [{"distribution": "bi_kappa", "kappa": 3.0, "temperature_parallel_eV": 260.0, "temperature_perpendicular_eV": 120.0, "magnetic_field_unit": [1.0, 0.0, 0.0], "flow_projection_m_s": 0.0}],
        "incident_direction": [1.0, 0.0, 0.0],
        "scattered_direction": [0.0, 1.0, 0.0]
    },
    "kappa_electrostatic_susceptibility": {
        "v": np.linspace(-5.0e7, 5.0e7, 2001).tolist(),
        "phase_velocity_m_s": 1.0e6,
        "electron_flow_projection_m_s": 1.5e5,
        "electron_density_m3": 2.0e18,
        "electron_temperature_eV": 250.0,
        "kappa": 4.0,
        "wavenumber_m_inverse": 120.0,
        "minimum_velocity_grid_probability_coverage": 0.999
    },
    "thomson_full_temperature_tensor_spectral_fit": {
        "wavelength_nm": np.linspace(529.0, 535.0, 181).tolist(),
        "measured_spectra": [
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,800.0)).tolist(),
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,420.0)).tolist(),
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,260.0)).tolist(),
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,610.0)).tolist(),
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,530.0)).tolist(),
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,340.0)).tolist(),
            (0.03 + 2.0*_example_bimaxwellian_spectrum(np.linspace(529.0,535.0,181),532.0,np.pi/2.0,493.3333333333)).tolist()
        ],
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": 1.5707963267948966,
        "scattering_vector_unit": [[1,0,0],[0,1,0],[0,0,1],[0.7071067812,0.7071067812,0],[0.7071067812,0,0.7071067812],[0,0.7071067812,0.7071067812],[0.5773502692,-0.5773502692,0.5773502692]],
        "instrument_response_matrix": np.eye(181).tolist(),
        "initial_temperature_tensor_eV": [[600,0,0],[0,600,0],[0,0,600]],
        "maximum_projected_rms_beta": 0.1,
        "maximum_flow_beta": 0.1
    },
    "thomson_full_temperature_tensor": {
        "projected_temperature_eV": [800.0, 420.0, 260.0, 700.0, 495.0, 395.0, 373.3333333333],
        "scattering_vector_unit": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.7071067812,0.7071067812,0.0],[0.7071067812,0.0,0.7071067812],[0.0,0.7071067812,0.7071067812],[0.5773502692,-0.5773502692,0.5773502692]],
        "projected_temperature_covariance_eV2": (0.04*np.eye(7)).tolist(),
        "maximum_geometry_condition_number": 1000.0,
        "minimum_consistency_p_value": 0.001
    },
    "thomson_three_dimensional_flow": {
        "velocity_projection_m_s": [100000.0, 200000.0, 300000.0],
        "scattering_vector_unit": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "projection_covariance_m2_s2": [[1000000.0, 0.0, 0.0], [0.0, 1000000.0, 0.0], [0.0, 0.0, 1000000.0]],
        "scattering_vector_covariance": (1.0e-6*np.eye(9)).tolist(),
        "maximum_geometry_condition_number": 1000.0,
        "minimum_projection_consistency_p_value": 0.001
    },
    "gyrotropic_bi_maxwellian_anisotropy": {
        "projected_temperature_eV": [180.0, 105.0, 80.0],
        "magnetic_pitch_cosine": [1.0, 0.5, 0.0],
        "projected_temperature_covariance_eV2": [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]],
        "magnetic_pitch_cosine_covariance": [[0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]]
    },
    "gyrotropic_bi_maxwellian_spectral_fit": {
        "wavelength_nm": np.linspace(412.0, 652.0, 181).tolist(),
        "measured_spectra": [
            (0.02 + 0.8*_example_bimaxwellian_spectrum(np.linspace(412.0, 652.0, 181), 532.0, np.pi/2.0, 800.0)).tolist(),
            (0.02 + 0.8*_example_bimaxwellian_spectrum(np.linspace(412.0, 652.0, 181), 532.0, np.pi/2.0, 425.0)).tolist(),
            (0.02 + 0.8*_example_bimaxwellian_spectrum(np.linspace(412.0, 652.0, 181), 532.0, np.pi/2.0, 300.0)).tolist()
        ],
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": 1.5707963267948966,
        "magnetic_field_unit": [0.0, 0.0, 1.0],
        "scattering_vector_unit": [[0.0, 0.0, 1.0], [0.8660254038, 0.0, 0.5], [1.0, 0.0, 0.0]],
        "instrument_response_matrix": np.eye(181).tolist(),
        "initial_temperature_parallel_eV": 800.0,
        "initial_temperature_perpendicular_eV": 300.0,
        "temperature_bounds_eV": [10.0, 5000.0],
        "flow_bounds_m_s": [-1000000.0, 1000000.0],
        "maximum_projected_rms_beta": 0.1,
        "scattering_vector_covariance": None,
        "projection_geometry_cross_covariance": None,
        "maximum_geometry_condition_number": 1000.0,
        "minimum_projection_consistency_p_value": 0.001
    },
    "gyrotropic_bi_kappa_spectral_fit": {
        "wavelength_nm": np.linspace(412.0, 652.0, 181).tolist(),
        "measured_spectra": [
            (3.0 + 2.5e3*_example_kappa_bin_density(np.linspace(412.0, 652.0, 181), 532.0, 2.0*np.sin(np.pi/4.0), 0.0, 800.0, 4.0)).tolist(),
            (3.0 + 2.5e3*_example_kappa_bin_density(np.linspace(412.0, 652.0, 181), 532.0, 2.0*np.sin(np.pi/4.0), 0.0, 425.0, 4.0)).tolist(),
            (3.0 + 2.5e3*_example_kappa_bin_density(np.linspace(412.0, 652.0, 181), 532.0, 2.0*np.sin(np.pi/4.0), 0.0, 300.0, 4.0)).tolist()
        ],
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": 1.5707963267948966,
        "magnetic_field_unit": [0.0, 0.0, 1.0],
        "scattering_vector_unit": [[0.0, 0.0, 1.0], [0.8660254038, 0.0, 0.5], [1.0, 0.0, 0.0]],
        "instrument_response_matrix": np.eye(181).tolist(),
        "initial_temperature_parallel_eV": 800.0,
        "initial_temperature_perpendicular_eV": 350.0,
        "initial_kappa": 4.0,
        "kappa_bounds": [1.5001, 100.0],
        "temperature_bounds_eV": [10.0, 5000.0],
        "flow_bounds_m_s": [-1000000.0, 1000000.0],
        "maximum_projected_rms_beta": 0.1,
        "scattering_vector_covariance": None,
        "projection_geometry_cross_covariance": None,
        "maximum_geometry_condition_number": 1000.0,
        "minimum_projection_consistency_p_value": 0.001
    },
    "landau_fdt_cts": {
        "angular_frequency_rad_s": [1.0e5, 2.0e5],
        "normalized_velocity_x": [-4.0, 0.0, 4.0],
        "normalized_velocity_y": [-4.0, 0.0, 4.0],
        "normalized_velocity_z": [-4.0, 0.0, 4.0],
        "species": [
            {"name": "electrons", "density_m3": 1.0e18, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "ions", "density_m3": 1.0e18, "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,  # 100 eV in joules
        "wave_number_m_inverse": 0.001,
        "coulomb_logarithm": 12.0
    },
    "landau_fdt_full_em_constitutive": {
        "angular_frequency_rad_s": [1.0e5, 2.0e5],
        "normalized_velocity_x": [-4.0, 0.0, 4.0],
        "normalized_velocity_y": [-4.0, 0.0, 4.0],
        "normalized_velocity_z": [-4.0, 0.0, 4.0],
        "species": [
            {"name": "electrons", "density_m3": 1.0e18, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "ions", "density_m3": 1.0e18, "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,
        "wave_number_m_inverse": 0.001,
        "propagation_direction": [1.0, 0.0, 0.0],
        "common_bulk_flow_m_s": [0.0, 0.0, 0.0],
        "coulomb_logarithm": 12.0
    },
    "landau_fdt_full_em_response": {
        "angular_frequency_rad_s": [1.0e5, 2.0e5],
        "normalized_velocity_x": [-4.0, 0.0, 4.0],
        "normalized_velocity_y": [-4.0, 0.0, 4.0],
        "normalized_velocity_z": [-4.0, 0.0, 4.0],
        "species": [
            {"name": "electrons", "density_m3": 1.0e18, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "ions", "density_m3": 1.0e18, "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,
        "wave_number_m_inverse": 0.001,
        "propagation_direction": [1.0, 0.0, 0.0],
        "common_bulk_flow_m_s": [0.0, 0.0, 0.0],
        "coulomb_logarithm": 12.0
    },
    "polarized_kinetic_ece": {
        "frequency_Hz": [45.0e9],
        "segment_length_m": [0.001],
        "wavevector_rad_m": [[0.0, 0.0, 942.0]],
        "magnetic_field_T": [[0.0, 0.0, 2.0]],
        "electron_density_m3": [1.0e19],
        "electron_temperature_eV": [1000.0],
        "mode_electric_field_unit_flux": {"real": [[[27.449, 0.0], [0.0, 27.449], [0.0, 0.0]]], "imag": [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]},
        "dielectric_z_m": [0.0, 0.001],
        "dielectric_tensor_r": {"real": [[[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]], "imag": [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]},
        "harmonic_order": 2,
        "quadrature_order": 16,
        "damping_rate_rad_s": 2.0e7
    },
    "stratified_nonparaxial_vector": {
        "x_m": [-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008],
        "y_m": [-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008],
        "electric_field_V_m": {"real": [[[0,0,0] for _ in range(8)] for _ in range(8)], "imag": [[[0,0,0] for _ in range(8)] for _ in range(8)]},
        "z_m": [0.0, 0.001],
        "epsilon_r": {"real": [[[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]], "imag": [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]},
        "frequency_Hz": 10.0e9
    },
    "stratified_mode_conversion": {
        "omega_rad_s": 6.283185307179586e10,
        "z_m": [0.0, 0.001],
        "epsilon_r": {"real": [[[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]], "imag": [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]},
        "kx_rad_m": 0.0,
        "ky_rad_m": 0.0
    },
    "cts_conserving_bgk_fdt": {
        "angular_frequency_rad_s": [1.0e10, 2.0e10],
        "wave_number_m_inverse": 200000.0,
        "species": [
            {"name": "electrons", "density_m3": 1.0e19, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "deuterium", "density_m3": 1.0e19, "mass_kg": 3.3435837724e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,
        "collision_frequencies_s_inverse": [2.0e7, 1.0e5],
        "bulk_flow_m_s": 0.0,
        "quadrature_order": 64,
        "quadrature_max_order": 384,
        "quadrature_convergence_tolerance": 1.0e-5
    },
}
 
# Reuse the same explicit physical input contract for the public response alias.
EXAMPLES["landau_fdt_full_em_response"] = dict(EXAMPLES["landau_fdt_full_em_constitutive"])

# A deterministic synthetic measured-spectrum case for the joint spatial inverse.
_joint_ts_wavelength = np.linspace(520.0, 544.0, 121)
_joint_ts_radius = np.array([0.0, 0.5, 1.0])
_joint_ts_temperature = np.array([80.0, 100.0, 120.0])
_joint_ts_flow = np.array([-5.0e3, 0.0, 5.0e3])
_joint_ts_sigma = 2.0 * 532.0 * np.sqrt(_joint_ts_temperature / 510998.95069) * np.sin(np.pi / 4.0)
_joint_ts_center = 532.0 * (1.0 - 2.0 * np.sin(np.pi / 4.0) * _joint_ts_flow / 299792458.0)
_joint_ts_spectra = (
    np.exp(-0.5 * ((_joint_ts_wavelength[None, :] - _joint_ts_center[:, None]) / _joint_ts_sigma[:, None]) ** 2)
    + 0.01
)
EXAMPLES["thomson_joint_spatial_spectra"] = {
    "wavelength_nm": _joint_ts_wavelength.tolist(),
    "measured_spectra": _joint_ts_spectra.tolist(),
    "radius_m": _joint_ts_radius.tolist(),
    "spatial_mixing_matrix": np.eye(3).tolist(),
    "spatial_mixing_matrix_covariance": None,
    "spectral_mixing_cross_covariance": None,
    "laser_wavelength_nm": 532.0,
    "scattering_angle_rad": float(np.pi / 2.0),
    "instrument_response": np.eye(_joint_ts_wavelength.size).tolist(),
    "signal_uncertainty": np.full_like(_joint_ts_spectra, 0.01).tolist(),
    "systematic_covariance": None,
    "regularization_strength": 0.0,
    "regularization_kind": "curvature",
    "tv_huber_scale": 0.05,
    "physical_model": "maxwellian",
}


def decode_complex_json(value):
    """Decode nested JSON arrays; {real: ..., imag: ...} denotes complex data."""
    if isinstance(value, Mapping):
        if set(value) == {"real", "imag"}:
            real = np.asarray(decode_complex_json(value["real"]), dtype=float)
            imag = np.asarray(decode_complex_json(value["imag"]), dtype=float)
            if real.shape != imag.shape:
                raise ValueError("Complex JSON real and imag arrays must have identical shapes.")
            return real + 1j * imag
        return {key: decode_complex_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_complex_json(item) for item in value]
    return value


def json_safe(value):
    """Convert numerical solver outputs to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items() if key != "landau_system"}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Sparse matrices and internal solver objects are deliberately not expanded.
    return str(value)


def render_advanced_thomson_physics_page(st) -> None:
    """Render the explicit advanced Analysis-1 solver interface."""
    st.subheader("Analysis 1 — advanced kinetic and electromagnetic physics")
    st.info(
        "These are explicit numerical solvers, not automatic fits to an arbitrary uploaded file. "
        "Supply calibrated/prepared physical inputs and geometry. No calibration, equilibrium "
        "reconstruction, or missing-input estimation is performed."
    )
    model = st.selectbox(
        "Select the physical solver",
        list(MODEL_LABELS),
        format_func=lambda key: MODEL_LABELS[key],
        key="ts_advanced_physics_model",
    )
    if model in MODEL_PHYSICS_NOTES:
        st.info(MODEL_PHYSICS_NOTES[model])
    st.caption(
        "For joint spatial Thomson inversion, systematic_covariance accepts a full channel-major covariance matrix for independent calibration/systematic errors; it is added to measurement and model-discrepancy covariance, not substituted for either."
    )
    st.caption(
        "For joint spatial Thomson inversion, regularization_kind accepts curvature or huber_total_variation. "
        "The latter is an edge-preserving Huber-TV prior on normalized radius; tv_huber_scale controls its dimensionless quadratic-to-linear gradient transition. "
        "It does not add measurement information or resolve data-unidentifiable modes."
    )
    st.caption(
        "Enter one JSON object containing the keyword arguments for the selected solver. "
        "For complex-valued arrays use {\"real\": [...], \"imag\": [...]}; both arrays must have identical shapes. "
        "The example is an input-shape template, not experimental data or a recommended physical operating point."
    )
    raw = st.text_area(
        "Prepared physical inputs (JSON)",
        value=json.dumps(EXAMPLES[model], indent=2),
        height=360,
        key=f"ts_advanced_physics_config_{model}",
    )
    if st.button("Run selected advanced physics solver", type="primary", key="run_ts_advanced_physics"):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("The solver configuration must be a JSON object.")
            config = decode_complex_json(parsed)
            result = run_advanced_thomson_physics(model, **config)
            safe = json_safe(result)
            if isinstance(result, Mapping) and result.get("primary_result_status") == "identified_from_measurement":
                st.success("Numerical solver completed; the primary physical result passed the solver data-identifiability gate.")
            elif isinstance(result, Mapping) and (
                result.get("primary_result_status") == "not_fully_identifiable_from_measured_spectrum"
                or result.get("status") == "conditional_regularized_candidate"
            ):
                st.warning(
                    "Numerical optimization completed, but the physical result is conditional: "
                    "the measured data do not identify all requested degrees of freedom. "
                    "Inspect the regularized candidate, rank, conditioning, and limitations; "
                    "do not report it as a uniquely measured EVDF."
                )
            else:
                st.success("The selected numerical solver completed. Review its limitations and convergence gates before interpreting results.")
            if isinstance(result, Mapping) and "captured_spectral_probability" in result:
                coverage = np.asarray(result["captured_spectral_probability"], dtype=float).ravel()
                st.subheader("Measured wavelength-window coverage")
                st.write([
                    {"view": index + 1, "captured_probability": float(value),
                     "missing_tail_probability": float(max(0.0, 1.0 - value))}
                    for index, value in enumerate(coverage)
                ])
                if result.get("spectral_window_coverage_warning"):
                    st.warning(result["spectral_window_coverage_warning"])
            if isinstance(result, Mapping) and "validity_screening" in result:
                st.subheader("Non-relativistic validity screen")
                st.json(json_safe(result["validity_screening"]))
            if isinstance(result, Mapping) and "relativistic_tail_diagnostic" in result:
                tail = result["relativistic_tail_diagnostic"]
                st.subheader("Projected-distribution relativistic-tail validity")
                st.latex(
                    r"P(|v_q|\geq c)=F_{t_\nu}((-c-u_q)/s)"
                    r"+1-F_{t_\nu}((c-u_q)/s),\quad \nu=2\kappa-1"
                )
                st.json(json_safe(tail))
                if tail.get("requires_relativistic_forward_model", False):
                    st.warning(
                        "The fitted non-relativistic distribution assigns an excessive "
                        "projected probability to superluminal velocities. Use the "
                        "relativistic projected-EVDF route before interpreting this fit physically."
                    )
            if isinstance(result, Mapping) and "velocity_moment_covariance_order" in result:
                st.subheader("Projected EVDF — physical velocity moments")
                st.latex(
                    r"T_{e,q}=\frac{m_e}{e k_B}"
                    r"\left[\langle v_q^2\rangle-\langle v_q\rangle^2\right]"
                )
                st.write({
                    "mean_velocity_parallel_m_s": result.get("mean_velocity_parallel_m_s"),
                    "velocity_parallel_standard_deviation_m_s": result.get("velocity_parallel_standard_deviation_m_s"),
                    "directional_equivalent_temperature_eV": result.get("directional_equivalent_temperature_eV"),
                    "skewness": result.get("skewness"),
                    "excess_kurtosis": result.get("excess_kurtosis"),
                    "uncertainty_status": result.get("velocity_moment_uncertainty_status"),
                    "standard_uncertainties": result.get("velocity_moment_standard_uncertainty"),
                })
                moment_covariance = result.get("velocity_moment_covariance")
                if moment_covariance is not None:
                    st.caption(
                        "Rows/columns follow velocity_moment_covariance_order; off-diagonal terms "
                        "retain correlations between flow, variance, temperature and shape moments."
                    )
                    st.dataframe(
                        np.asarray(moment_covariance, dtype=float),
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "Physical-moment uncertainty is unavailable because the projected-EVDF "
                        "covariance did not pass its numerical availability gate."
                    )
            if isinstance(result, Mapping) and "fourth_moment_exists" in result:
                st.subheader("Kappa physical moment hierarchy")
                st.latex(r"\mathrm{Var}(v)=\frac{k_B T_e}{m_e}")
                st.write({
                    "second_moment_m2_s2": result.get("second_moment_m2_s2"),
                    "velocity_grid_probability_coverage": result.get("velocity_grid_probability_coverage"),
                    "fourth_moment_exists": result.get("fourth_moment_exists"),
                    "kurtosis": result.get("kurtosis"),
                    "excess_kurtosis": result.get("excess_kurtosis"),
                    "fourth_central_moment_m4_s4": result.get("fourth_central_moment_m4_s4"),
                })
                if not result["fourth_moment_exists"]:
                    st.warning(
                        "For this Kappa index (kappa <= 5/2), the fourth velocity moment "
                        "diverges. Kurtosis and fourth-moment-based diagnostics are therefore "
                        "undefined, not zero."
                    )
                st.caption(
                    "The analytic PDF remains normalized on the full velocity axis. "
                    "The grid-conditioned PDF is a visualization-only conditional density; "
                    "it is not substituted into the Landau susceptibility."
                )
            if isinstance(result, Mapping) and "three_dimensional_flow_inference" in result:
                flow_result = result["three_dimensional_flow_inference"]
                st.subheader("Joint 3-D electron-flow inference")
                st.latex(r"u_{q,i}=\hat{\mathbf q}_i\cdot\mathbf u_e")
                st.write({
                    "flow_projection_covariance_status": result.get("flow_projection_covariance_status"),
                    "identifiability": flow_result.get("identifiability"),
                    "reportable_as_inferred_parameter": flow_result.get("reportable_as_inferred_parameter"),
                    "electron_velocity_vector_m_s": flow_result.get("electron_velocity_vector_m_s"),
                    "electron_velocity_covariance_m2_s2": flow_result.get("electron_velocity_covariance_m2_s2"),
                    "reason": flow_result.get("reason"),
                })
                if not flow_result.get("reportable_as_inferred_parameter", False):
                    st.warning(
                        "The 3-D flow vector is withheld unless the view directions and "
                        "fit-derived cross-view flow covariance identify all three components."
                    )
            if isinstance(result, Mapping) and result.get("electron_pressure_tensor_Pa") is not None:
                st.subheader("Electron pressure tensor — density-conditioned closure")
                st.latex(r"\mathbf P_e = n_e e\,\mathbf T_{e,\mathrm{eV}}")
                st.write({
                    "electron_density_m3": result.get("electron_density_m3"),
                    "electron_density_uncertainty_m3": result.get("electron_density_uncertainty_m3"),
                    "electron_pressure_tensor_Pa": result.get("electron_pressure_tensor_Pa"),
                    "pressure_parameter_covariance_Pa2": result.get("electron_pressure_parameter_covariance_Pa2"),
                    "covariance_provenance": result.get("electron_pressure_covariance_status"),
                })
                if result.get("electron_pressure_parameter_covariance_Pa2") is None:
                    st.warning("Pressure is derived, but its covariance is withheld because the temperature-tensor covariance is unavailable at the active PSD boundary or was not estimable.")
            if isinstance(result, Mapping) and "thermal_velocity_covariance_tensor_per_view_m2_s2" in result:
                st.subheader("Gyrotropic 3-D thermal velocity covariance")
                st.latex(
                    r"\langle\delta\mathbf v\,\delta\mathbf v^T\rangle"
                    r"=\frac{e}{m_e}\left[T_\perp I+(T_\parallel-T_\perp)"
                    r"\hat{\mathbf b}\hat{\mathbf b}^T\right]"
                )
                st.write({
                    "temperature_anisotropy_ratio_parallel_over_perpendicular":
                        result.get("temperature_anisotropy_ratio_parallel_over_perpendicular"),
                    "covariance_tensor_per_view_m2_s2":
                        result.get("thermal_velocity_covariance_tensor_per_view_m2_s2"),
                    "projected_variance_from_tensor_m2_s2":
                        result.get("projected_velocity_variance_from_tensor_m2_s2"),
                    "projected_variance_expected_from_fit_m2_s2":
                        result.get("projected_velocity_variance_expected_m2_s2"),
                    "projected_variance_relative_residual":
                        result.get("projected_velocity_variance_relative_residual"),
                    "definition": result.get("thermal_velocity_covariance_definition"),
                })
                if result.get("thermal_velocity_covariance_tensor_per_view_m2_s2") is None:
                    st.warning(
                        "The 3-D thermal covariance tensor is withheld because both "
                        "temperature components are not both identifiable within the declared model-validity domain."
                    )
            if isinstance(result, Mapping) and "electron_temperature_tensor_eV" in result:
                st.subheader("General non-gyrotropic 3-D electron temperature tensor")
                st.latex(r"T_q=\hat{\mathbf q}^{T}\mathbf T_e\hat{\mathbf q},\qquad \mathbf T_e=\mathbf L\mathbf L^T")
                st.write({
                    "temperature_tensor_eV": result.get("electron_temperature_tensor_eV"),
                    "principal_temperature_eigenvalues_eV": result.get("principal_temperature_eigenvalues_eV"),
                    "principal_axes_cartesian_columns": result.get("principal_axes_cartesian_columns"),
                    "projected_temperature_eV": result.get("projected_temperature_eV"),
                    "view_dyad_rank": result.get("view_dyad_rank"),
                    "spectral_jacobian_rank": result.get("jacobian_rank"),
                    "spectral_jacobian_condition_number": result.get("jacobian_condition_number"),
                    "tensor_covariance_eV2": result.get("temperature_tensor_parameter_covariance_eV2"),
                    "model_validity_status": result.get("model_validity_status"),
                })
                if not result.get("reportable_as_inferred_parameter", False):
                    st.warning("The tensor is not reportable as a physical inference outside the declared non-relativistic validity domain.")
            st.json(safe)
            st.download_button(
                "Export advanced physics result (JSON)",
                data=json.dumps(safe, indent=2, allow_nan=False),
                file_name=f"tis_{model}_result.json",
                mime="application/json",
                key="download_ts_advanced_physics_result",
            )
        except Exception as exc:
            st.error(f"Advanced physics solver failed: {exc}")
