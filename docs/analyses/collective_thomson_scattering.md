# Analysis 1 subanalysis — Microwave Collective Thomson Scattering (CTS)

## Scope and model selection

Microwave CTS is a dedicated subanalysis within primary Analysis 1. It is physically distinct from optical Thomson scattering and retains its own model selection and execution contracts. Its public dispatcher is:

- \`tokamak_analysis.cts_analysis.analyze_cts_spectrum_model(model="electrostatic", ...)\`
- \`tokamak_analysis.cts_analysis.analyze_cts_spectrum_model(model="full_electromagnetic", ...)\`

The CTS model must be selected explicitly. The software never silently substitutes the electrostatic model for the magnetized full-EM model or vice versa, and neither CTS model is treated as an optical Thomson model.

The public dispatcher uses one shared keyword contract for both branches: pass the calibrated measured power vector as `spectrum`. For the Full-EM branch, the dispatcher maps it to the internal `measured_spectrum` parameter. Full-EM incident and scattered beam directions are fields inside the `scattering_geometry` mapping (or its frequency-resolved geometry records), not separate keyword arguments. The optional `mode_index` belongs to the ray-tracing helper that creates frequency-resolved geometry; it is not an argument to the spectrum-inversion dispatcher.

The electrostatic and Full-EM CTS analyses require the supplied calibrated measured response matrix to be finite, non-negative, square on the measured frequency grid, and already unit-sum by column. The base measured operator is used unchanged: TIS rejects a non-normalized matrix instead of renormalizing it. The Full-EM dashboard also accepts an optional measured response-matrix 1σ uncertainty CSV and an explicit Monte Carlo refit count. In that separate uncertainty-propagation path, perturbed response realizations are projected back to the column-normalized non-negative response-operator constraint before refitting; this does not alter the supplied base measurement operator, calibrate the instrument, or validate the response matrix. The electrostatic and Full-EM response uncertainty representations are model-specific and must not be interchanged.

All spectra and frequency axes are assumed to be measured, prepared and calibrated. The CTS workflows do not calibrate the instrument, correct raw detector counts, infer a missing frequency axis, or perform experimental validation.

## Electrostatic longitudinal CTS

The electrostatic path remains an unmagnetized multi-species longitudinal dielectric model. It computes species susceptibilities, the longitudinal dielectric function, the dynamic structure factor and convolution with the supplied measured frequency-response matrix.

It supports the declared Maxwellian background and fast-ion options, including a one-dimensional projected tabulated fast-ion VDF. Ion species, charge states, masses and charge-density fractions are explicit external inputs. The inversion fits electron-flow and a shared projected ion-flow component from the measured CTS spectrum, while independently fitting one Maxwellian ion temperature per supplied species. `ion_temperature_initial_eV` accepts either a scalar seed (broadcast to all species for backward compatibility) or a vector with one seed per species. `parameter_bounds` accepts either a shared `ion_temperature_eV` pair or individual `ion_temperature_eV_0`, `ion_temperature_eV_1`, … pairs; the two forms cannot be mixed. Species-specific ion-temperature uncertainties and their covariance are returned. The common projected ion flow remains shared because independent species flow projections are not generally identifiable from a single longitudinal spectrum without additional constraints. For frequency-resolved geometry, each measured bin is evaluated with its own supplied scattering vector. The flow-support bound uses the maximum wave-number magnitude across all supplied bins; automatic width-to-temperature optimizer seeds are rejected because a single-q width conversion is not defined for a varying geometry. Supply explicit density, electron-temperature and ion-temperature optimizer seeds for that case. Density and temperature optimizer seeds are optional: omitted values are initialized automatically from measured spectral shape and width, while the final parameters are determined by the full forward-model fit. `electron_velocity_projection_m_s` and `ion_velocity_initial_m_s` are optional numerical optimizer seeds; omission uses zero as a numerical starting value only. Neither is a fixed measured plasma parameter or a stationary-flow assumption. The dashboard labels the electron-flow value as an optional seed and no longer requires the user to enter zero to assert stationary flow.
The electrostatic dashboard also accepts an optional joint external-parameter covariance CSV with an explicit ordered parameter list. Supported names are `probe_frequency_Hz`, `scattering_angle_rad`, and `ion_fraction_i` (zero-based species index). This covariance is mutually exclusive with the separate scalar uncertainty fields; the uploaded matrix is propagated through correlated Monte Carlo refits. The covariance is user-supplied and is not inferred from the spectrum.

## Magnetized full-electromagnetic CTS

The full-EM path is implemented in \`tokamak_analysis.cts_full_em\`. It combines:

1. Magnetized Maxwellian kinetic conductivity and screened density/current/electric/magnetic fluctuation covariance, retaining all supplied auto- and cross-correlations.
2. The low-temperature Bindslev \`Gamma^{-1}\` / \`X\` source-current closure for Maxwellian bulk-electron three-wave coupling.
3. Species-resolved thermal-ion inference. Each supplied ion species has its own fitted temperature and flow projection along the scattering wavevector; ion composition remains a fixed external input.
4. Optional relativistic gyrotropic electron and fast-ion populations are supported by the forward spectral-density function. The ordinary measured-spectrum plasma-parameter inversion rejects non-null `electron_distribution`, because it does not jointly fit electron flow with that explicit relativistic VDF. A separate `fit_full_em_gyrotropic_population_shape` workflow performs constrained VDF-shape inference with bulk plasma parameters and other listed inputs fixed; it is not an unrestricted reconstruction of an arbitrary three-dimensional EVDF. The dashboard therefore does not offer the forward-only VDF input as an inversion control. Explicit relativistic gyrotropic electron distributions are available to the forward spectral-density API and the separate constrained population-shape inversion, not to the ordinary plasma-parameter inversion. Fast-ion population shapes supplied to Full-EM inversion likewise remain fixed external inputs. Fast-ion populations, when supplied to inversion, remain fixed external distributions rather than fitted VDFs.
5. Optional frequency-resolved cold-plasma ray tracing and path-integrated Gaussian beam-overlap weighting.

### Relativistic kinetic populations and three-wave vertex

Implemented in \`tokamak_analysis.cts_relativistic\`:

- Relativistic Maxwell-Juttner momentum distribution for a thermal relativistic electron population. Its temperature follows the fitted electron temperature when \`electron_distribution={"kind":"relativistic_maxwell_juttner"}\` is supplied without an explicit temperature.
- Relativistic anisotropic bi-Gaussian momentum distributions with explicit perpendicular and parallel temperatures.
- Caller-supplied tabulated gyrotropic distributions \(f(p_\perp,p_\parallel)\).
- Relativistic linear Vlasov response using \(v=p/(\gamma m)\) and \(\Omega/\gamma\).
- Klimontovich particle shot-noise covariance for the supplied relativistic populations.
- A second-order relativistic Vlasov three-wave current vertex, evaluated from the first-order perturbations and momentum derivatives of the supplied VDF.

When an electron distribution is explicitly supplied to the forward model, the Maxwellian electron response and the Bindslev low-temperature electron nonlinear source are replaced by the relativistic kinetic electron response and second-order vertex. When no electron distribution is supplied, the Maxwellian bulk-electron Bindslev closure remains active. The dimensionless theta_e = T_e/(m_e c²) is reported. In ordinary measured-spectrum plasma-parameter inversion, `bindslev_validity_theta_e_limit` applies only to the Maxwellian Bindslev closure; explicit relativistic electron input is rejected there. The separate population-shape inversion uses its own constrained kinetic model. If no screening limit is supplied, the Maxwellian result is marked `not_quantitatively_screened`; the software does not invent a universal cutoff.

The relativistic vertex is a homogeneous local kinetic model with prescribed gyrotropic distributions and Lorentzian resonance broadening. It does not include finite-orbit-width effects, collisions, spatially varying relativistic kinetic response or nonlinear equilibrium evolution. The relativistic distribution shape is supplied by the researcher; the software does not reconstruct an arbitrary multidimensional VDF from the CTS spectrum.

### Ray tracing and scattering-volume integration

Implemented in \`tokamak_analysis.cts_em_geometry.trace_cts_frequency_resolved_beams\` and connected to \`full_em_cts_spectral_density\`:

- Cold-plasma Stix dielectric tensor and Maxwell wave branches.
- Hamiltonian geometrical-optics tracing of the incident beam and a receiver ray for each scattered-frequency bin.
- Local Gaussian beam-intensity overlap evaluated between paired ray samples.
- Path-length quadrature along the incident ray, with normalized weights proportional to local Gaussian overlap times path length.
- Full-EM spectral evaluation at each returned scattering-volume sample and weighted summation of the local spectra.

The equilibrium callback must supply the local magnetic field and all species density profiles. The ray geometry JSON must preserve the exact measured \`frequency_shift_Hz\` axis, record the explicitly selected \`mode_index\` (0, 1 or 2), and contain one \`frequency_resolved_geometry\` record per measured bin, with \`scattering_samples\` and non-negative \`sample_weight\` values summing to one. The ray helper has no default propagation branch.

**Declared geometry limitations:** this is cold-plasma geometrical optics, not diffraction-resolved quasi-optical propagation. Beam covariance matrices are supplied along the rays; the software does not propagate them. The fitted electron density, electron temperature and thermal-ion temperatures are assumed uniform across the sampled scattering volume, while local magnetic field and ray directions may vary. Absolute beam-overlap path integrals are used as relative frequency-bin weights; absolute launcher/receiver power coupling is not calibrated or inferred. Mode conversion and branch tracking through degeneracies are not implemented.

## Automatic numerical convergence assessment

Every Full-EM measured-spectrum inversion now performs a paired harmonic-order / velocity-quadrature refinement study by default after the best-fit physical parameters have been obtained. The default levels begin at the inversion's selected `harmonic_order` and `quadrature_order`, then increase each by two documented increments; the default relative L2 and L-infinity tolerance is `1e-3`. Researchers may supply explicit paired refinement sequences and a different positive tolerance, or explicitly disable the assessment with `numerical_convergence=False`. The returned `numerical_convergence` record contains each level, both relative changes, the final refined spectrum and a pass/fail/not-assessed status. This is a numerical discretization check at fixed fitted parameters; it is not a physical uncertainty, optimizer-identifiability test, model-discrepancy estimate or experimental validation. A failed convergence check does not silently modify the fitted physical parameters.

## Full-EM input contract

Required inputs include:

- Calibrated frequency-shift axis and calibrated CTS spectral power.
- Probe frequency and measured instrument-response matrix.
- Optional researcher-supplied initial electron density, electron temperature and ion-temperature estimates. Any omitted seed is estimated automatically from the calibrated measured spectrum: electron temperature from the full-spectrum RMS width, common ion temperature from the central 68%-weight core width, and density from an explicitly labelled alpha=1 numerical seed. These are optimizer starts only, not reported measurements; supplied values override the corresponding automatic estimates.
- Explicit ion species metadata and charge-density fractions.
- Optional researcher-supplied parameter bounds keyed by electron density, electron temperature and species-specific ion temperatures. If omitted, the positive log-parameters are not restricted by default scientific ranges.
- Magnetic-field vector.
- Optional initial electron-flow projection seed and optional per-species ion-flow projection seeds along the scattering wavevector. The inversion fits these scalar projections from the measured spectrum; a single CTS geometry does not identify full three-dimensional flow vectors.
- Incident/scattered wave directions and polarization vectors.
- Measured spectral uncertainty or a full frequency-bin covariance matrix.

Optional inputs include:

- The ordinary Full-EM plasma-parameter inversion rejects explicit `electron_distribution`. Use the separate `fit_full_em_gyrotropic_population_shape` API for constrained VDF-shape inference with bulk parameters fixed; this is not an unrestricted three-dimensional EVDF reconstruction.
- Relativistic fast-ion population metadata.
- Positive-definite spectral covariance.
- Frequency-resolved ray-traced scattering-volume geometry.
- Explicit harmonic order, quadrature order and resonance damping.

Flow projections omitted from the inversion's optional seed inputs are initialized numerically at zero and then fitted; zero is not held fixed as a stationary-flow assumption. Ion composition, polarization, VDF shape and equilibrium profile are not silently inferred and must be supplied where required.

## Numerical convergence and uncertainty scope

The measured-spectrum inversion automatically calls `tokamak_analysis.cts_full_em.assess_full_em_cts_numerical_convergence` after fitting, unless the researcher explicitly sets `numerical_convergence=False`. It evaluates paired increasing `harmonic_order` and `quadrature_order` values while holding fitted physical parameters, external inputs and damping prescription fixed. It reports successive relative L2 and L-infinity spectral changes and a tolerance-based convergence status. The helper can also be called independently for a forward-model convergence study. This is a numerical discretization check only; it is not experimental validation or a physical-model uncertainty estimate.

Pointwise spectral uncertainty or a full frequency-bin covariance is optional for Full-EM inversion. If supplied, it weights the fit; if omitted, the fit is unweighted and the local Jacobian covariance is scaled by residual variance, explicitly not measurement-noise-calibrated. The result preserves this as `conditional_fit_covariance`. When measured response-matrix uncertainty is also supplied, its successful Monte Carlo refit covariance is exposed separately as `instrument_response_parameter_covariance` and in `external_uncertainty_covariance_components`; `fit_covariance` is then the sum of those two components. The combined scope explicitly excludes cross-covariances with other external inputs. The standalone local-input covariance sampler independently validates that its covariance is finite, real, symmetric, positive semidefinite and dimensionally consistent with the ordered base vector before drawing reproducible perturbations; it does not infer or repair an invalid covariance. It does not constitute a complete uncertainty budget for magnetic field, ion composition, beam directions, polarization, ray tracing, beam covariance propagation or supplied kinetic-distribution shape. Those fixed inputs and exclusions are exposed in the result provenance/scope fields. The damping default is a numerical resonance regularization, not a physical collision frequency.

## Inversion and uncertainty

The full-EM inversion uses least-squares in logarithmic positive parameters, profiles spectral amplitude and constant background as nuisance parameters, and fits electron density, electron temperature, the electron-flow projection, and one temperature plus one flow projection per supplied thermal-ion species. It now runs a deterministic seven-start local search: the researcher-supplied or automatically estimated seed plus symmetric log-space perturbations of density, electron temperature and the full species-resolved ion-temperature vector. The successful fit with the lowest weighted residual objective is selected, and per-start convergence/objective metadata are returned. This reduces dependence on one arbitrary local start but is not a proof of global optimality or a posterior multimodality analysis. The optional flow-projection inputs are optimizer seeds, not fixed flow vectors. No scientific parameter bounds are imposed by default; optional bounds must be supplied explicitly by the researcher. Species composition and supplied fast-ion VDF shapes remain external inputs. The ordinary Full-EM plasma-parameter inversion rejects explicit relativistic electron distributions. The separate `fit_full_em_gyrotropic_population_shape` workflow infers VDF shape while bulk ne/Te/Ti and other specified inputs remain fixed; it does not perform unrestricted simultaneous recovery of an arbitrary EVDF and bulk plasma parameters.

The local Jacobian covariance is reported with fit residuals, identifiability rank and condition number. Full spectral covariance is used for whitening when supplied. The full-EM path does not claim complete propagation of every uncertainty in the supplied equilibrium, ray geometry, beam covariance, VDF or numerical truncation. These are declared model/input uncertainties, not set to zero.

The electrostatic path retains its own differential-evolution, improved-genetic and regularized tabulated-VDF options. Those strategies are not silently applied to the full-EM solver.

## Recent peer-reviewed method basis (2024–2026)

The physical response model remains explicitly selected between the unmagnetized longitudinal electrostatic dielectric formulation and the magnetized full-electromagnetic kinetic formulation. Recent papers are used to guide bounded extensions and inversion methodology; they do not replace the underlying dielectric response equations or validate this implementation.

- **CTS background subspace methods:** T. Verdier et al., “Robust monitoring of thermal and fast ions using collective Thomson scattering: Combining physics- and data-driven background estimation,” *Review of Scientific Instruments* **96** (2025), 043513, DOI: 10.1063/5.0238948. The published method uses rapidly modulated on/off CTS measurements and a physics-informed PCA approach. TIS currently implements a narrower, explicitly supplied reference-spectrum PCA nuisance subspace for the electrostatic CTS inversion. It does not claim to reproduce the paper's modulation-aware physics-informed algorithm, and reference-ensemble uncertainty is not yet propagated.
- **CTS spectral inversion:** “Spectral decomposition for collective Thomson scattering based on an improved genetic algorithm,” *Review of Scientific Instruments* **95** (2024), 093542, DOI: 10.1063/5.0215734 The paper motivates a bounded global-search alternative and sensitivity-aware fitting. TIS's existing improved-genetic search is a separate implementation; no equivalence to the published fitness function is claimed.
- **Accuracy-driven CTS modelling and inversion:** Flocken, Korsholm & Rasmussen, “Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas,” *Fusion Engineering and Design* (2026), DOI: 10.1016/j.fusengdes.2026.115845. This recent work combines ray tracing, forward modelling and Bayesian inversion with explicit nuisance parameters and repeated fits. It motivates further uncertainty-aware full-EM inference; it is not itself a new dielectric response model.
- **Single-view fast-ion inversion:** Jiang et al., “Physically constrained data-driven inversion of 2D fast-ion velocity distributions from single view CTS spectra,” *Nuclear Fusion* **66** (2026), 056012, DOI: 10.1088/1741-4326/ae4fdc. This is a recent proposed route beyond the currently implemented projected 1-D electrostatic fast-ion VDF. A 2-D inversion must remain a separately selected, physically constrained inverse problem with explicit priors and identifiability reporting; it must not be implied by the existing 1-D API.

The electrostatic CTS genetic-search implementation now validates its controls, rejects an entirely non-finite objective population, preserves the best-so-far candidate, and returns a monotone convergence history including the initial and final populations. It uses the CTS forward-model residual objective; it is not represented as a reproduction of Zhang et al.'s paper-specific fitness function. The reference-PCA nuisance model and genetic-search implementation are not described as reproductions of the cited papers. No experimental validation of TIS is performed or claimed.

## Software tests and validation boundary

Regression tests cover electrostatic spectra, full-EM covariance structure, Bindslev tensors and correlated contraction, relativistic gyrotropic linear response and nonlinear vertices, species-resolved ion temperatures, cold-plasma ray tracing, Gaussian overlap, scattering-volume integration and explicit input validation.

These tests are synthetic numerical/software consistency checks. **No independent experimental validation is claimed or performed.**

## Scientific references

1. J. Sheffield, D. Froula, S. H. Glenzer and N. C. Luhmann Jr., *Plasma Scattering of Electromagnetic Radiation*, 2nd ed., Academic Press, 2011.
2. H. Bindslev, “Three Wave Mixing and Thomson Scattering in Plasmas,” JET-P(93)41, 1993.
3. H. Bindslev, “A quantitative study of scattering from electromagnetic fluctuations in plasmas,” *Journal of Atmospheric and Terrestrial Physics* **58** (1996), 983–989, DOI: [10.1016/0021-9169(95)00129-8](https://doi.org/10.1016/0021-9169(95)00129-8).
4. H. Bindslev, “Dielectric effects on Thomson scattering in a relativistic magnetized plasma,” *Plasma Physics and Controlled Fusion* **33** (1991), 1775–1804, DOI: [10.1088/0741-3335/33/14/005](https://doi.org/10.1088/0741-3335/33/14/005).
5. X. Fang et al., “Study on the Full Electromagnetic Model of Collective Thomson Scattering in Magnetic Confinement Fusion,” *Power Generation Technology* **46**(1), 200–208 (2025), DOI: [10.12096/j.2096-4528.pgt.24149](https://doi.org/10.12096/j.2096-4528.pgt.24149).
6. T. Verdier et al., “Robust monitoring of thermal and fast ions using collective Thomson scattering: Combining physics- and data-driven background estimation,” *Review of Scientific Instruments* **96** (2025), 043513, DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948).
7. “Quasi-optical beam tracing module development for millimeter-wave high-wavenumber collective scattering on the NSTX-U and EAST tokamaks,” *Fusion Engineering and Design* **212** (2025), 114826, DOI: [10.1016/j.fusengdes.2025.114826](https://doi.org/10.1016/j.fusengdes.2025.114826). This is a diffraction-aware reference; TIS currently implements geometrical optics with supplied local Gaussian covariance.
8. “Development and Application of Collective Thomson Scattering Diagnostics for Energetic and Bulk Ion Measurements in the Large Helical Device,” *Journal of Fusion Energy* **45** (2026), 55, DOI: [10.1007/s10894-026-00595-7](https://doi.org/10.1007/s10894-026-00595-7).
9. J. L. Flocken, S. B. Korsholm and J. Rasmussen, “Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas,” *Fusion Engineering and Design* (2026), 115845, DOI: [10.1016/j.fusengdes.2026.115845](https://doi.org/10.1016/j.fusengdes.2026.115845).


## Model-conditional inputs and numerical provenance

The public contract distinguishes Electrostatic and Full-EM inputs. The Electrostatic path alone consumes the scalar electron-flow projection, common projected ion-flow initial estimate, Electrostatic inversion-method/global-search controls, Electrostatic response-matrix Monte Carlo controls and its supported external-parameter uncertainty mapping. The Full-EM inversion consumes the magnetic-field vector, incident/scattered directions and complex polarizations, harmonic/quadrature orders, optional electron/ion flow-projection initial seeds, fixed fast-ion populations, frequency-resolved ray geometry and scattering-volume sample controls. Full three-dimensional flow vectors are accepted by the forward spectral-density function, not by the spectrum-inversion dispatcher. Explicit relativistic electron distributions are likewise a forward-model capability only and are rejected by the current inversion. Density/temperature initial estimates, ion species/composition, measured frequency axis/spectrum, probe frequency and measured response are shared. These conditions are encoded in `DIAGNOSTIC_REGISTRY.json` using `required_when_model`; a single flat list of mandatory inputs must not be used for both models.

When `damping_rate_rad_s` is omitted, Full-EM uses the documented numerical regularization

\[
\gamma = \max(2\pi f_0\times 10^{-7},\;1\;\mathrm{rad\,s^{-1}}).
\]

This is not a physical collision frequency. The resolved value and whether it was caller-supplied or defaulted are returned in `numerical_controls`. Harmonic order, quadrature order and damping should be varied in numerical convergence studies before interpreting the result. Ray-tracing APIs require an explicit `mode_index`; no propagation branch is silently selected.

If no pointwise uncertainty or spectral covariance is supplied to Electrostatic CTS, the fit is unweighted. The implementation does not invent a noise scale from the measured spectrum; the result reports `spectral_weighting_status=unweighted_no_measurement_uncertainty_supplied`, and reduced chi-square is unavailable.


## Automatic optimizer initialization

The electrostatic inversion does not require users to provide density or temperature starting guesses. For each omitted seed, the initializer uses the calibrated spectral power after subtracting a low-percentile baseline. The electron-temperature seed is derived from the full-spectrum RMS frequency width and the scattering wave number; the common ion-temperature seed uses the central 68%-weight spectral-core width and the charge-density-weighted ion mass. A density seed is constructed by setting the dimensionless collective parameter to alpha=1 solely for numerical initialization. This alpha value is not an inferred plasma measurement, a prior, or a constraint: the inversion fits density freely from the complete measured spectrum. Explicit researcher-supplied seeds override only their corresponding automatic estimates. The result records the initialization method and provenance separately from fitted physical outputs.

The initializer is deterministic and does not perform instrument calibration, experimental validation, or detector correction. It operates on the calibrated CTS spectrum supplied to the analysis.


## Cold-plasma eigenmode branch tracking

`tokamak_analysis.cts_em_geometry.track_cold_plasma_modes_from_operators`
tracks eigenpolarization branches along an ordered sequence of cold-plasma
wave operators using maximum squared polarization overlap, a global
Hungarian assignment, and phase alignment. It explicitly flags near-degenerate
eigenvalue subspaces, where an individual eigenvector is not uniquely
defined.
This is branch-continuity bookkeeping, not a coupled-mode conversion solver.
The method complements the ray/forward-modelling and Bayesian-inversion
workflow discussed in recent CTS diagnostic design work (e.g. 2026 accuracy-
driven CTS optimisation) and does not claim experimental validation.



The cold-plasma ray result now also includes the polarization-continuity
diagnostics from `track_cold_plasma_modes_from_operators`. It reports whether
the fixed eigenvalue-order index used by the Hamiltonian ODE remains the same
tracked polarization branch along the sampled ray. A branch permutation or
near-degenerate subspace is reported explicitly; the ray solver does not
silently claim coupled-mode conversion or continuity through a degeneracy.


## Degeneracy-aware coherent CTS mode conversion

The cold-plasma mode utility now has two distinct operations. 
`track_cold_plasma_modes_from_operators` performs branch bookkeeping and flags
degenerate eigenspaces. `propagate_cold_plasma_mode_coupling` propagates a
three-component coherent modal-amplitude vector along an explicitly supplied
ordered path using the lossless coupled-mode equation

```
d(a)/ds = -[U^H dU/ds + i diag(beta)] a
```

Here `U` is the parallel-transported local polarization frame and `beta`
contains the caller-supplied propagation constants in rad/m. Degenerate
subspaces are aligned as subspaces by a unitary Procrustes transport; the
individual basis labels within an exact degeneracy are not treated as
observable. The solver reports modal powers, conversion coupling, degeneracy
flags, the total conserved modal power, and the power contained specifically
in each detected degenerate subspace (with the tracked mode indices). The
subspace-power diagnostic is zero where no degenerate subspace is detected;
it is not a duplicate of total modal power. It also reports the numerical
power-conservation residual. It rejects missing or misaligned
path/propagation-constant inputs rather than inventing them.

This is a **conditional local cold-plasma coupled-mode model**, not a full-wave
solution. It requires the researcher to provide the physical path, local
Maxwell operators and propagation constants for the same ray. It does not
model absorption, collisions, reflection at cutoffs, resonant layers,
diffraction or nonlinear coupling. Numerical regression tests are synthetic
software checks only; no experimental validation is claimed.

The physical context is consistent with (i) J. De Vadder, J. De Jonghe and
R. Keppens, “Collisional damping of wave modes in ion–electron plasmas,”
*Journal of Plasma Physics* 90 (2024), DOI:
10.1017/S0022377824000680, which explicitly documents wave-mode crossings and
avoided crossings, and (ii) X. Fang et al., “Study on the Full Electromagnetic
Model of Collective Thomson Scattering in Magnetic Confinement Fusion,”
*Power Generation Technology* 46 (2025), DOI:
10.12096/j.2096-4528.pgt.24149, for the full-electromagnetic CTS context.
The coupled-mode evolution is an explicit mathematical propagation layer
requiring supplied path and propagation constants; it is not attributed as an
algorithm reproduced from either paper.


### Ray-tracing integration contract

`trace_cold_plasma_ray` accepts the optional pair
`initial_mode_amplitudes` and `propagation_constants_rad_m`. Supplying both
activates the coherent coupled-mode propagation on the ray's physical
arc-length coordinate. Supplying only one is an input-contract error. For
`trace_cts_frequency_resolved_beams`, the same pair may be included in the
`incident_ray` mapping and independently in each `receiver_rays` mapping.
The propagation-constant array must have shape `(n_ray_samples, 3)`, in
rad/m, and must correspond to the same ordered ray samples used to construct
the local wave operators. Initial amplitudes are a three-component complex
vector in the initial flux-normalized modal basis. Their squared magnitudes
are modal powers; the lossless model conserves their sum.

The optional calculation is not inferred from frequency, refractive index or
the eigenvalues of the wave operator. A researcher must supply the actual
propagation constants and initial modal amplitudes. The geometry routine
continues to return branch-continuity diagnostics whether or not coupled-mode
propagation is requested.


## Conditional local external-input covariance in Full-EM inversion

For local (non-ray-resolved) geometry, `analyze_full_em_cts_spectrum` accepts an optional ordered positive-semidefinite `external_input_covariance` with its exact `external_input_parameter_order`. Supported components are the three magnetic-field components, incident/scattered direction components, and real/imaginary components of the incident/scattered complex polarization vectors. The supplied joint covariance is sampled reproducibly and the complete physical parameter fit is repeated for each accepted draw. The returned `external_input_parameter_covariance` is a separate physical-parameter covariance contribution, with requested/successful draw counts and an explicit propagation status.

This route is conditional on the supplied local covariance and is deliberately rejected for `frequency_resolved_geometry`: ray, equilibrium, path-sample and beam-covariance uncertainty are not represented by this local-vector contract. Covariance contributions from the measured response and local external-input Monte Carlo routes are combined as separate components; cross-covariance between those distinct sources is not inferred. No geometry, polarization, magnetic field or instrument response is calibrated or modified.


## Calibrated reference-spectrum PCA background (electrostatic CTS)

The electrostatic analyze_cts_spectrum workflow optionally accepts
background_reference_spectra and background_components. The reference
array contains at least two already calibrated background-only spectra,
sampled on exactly the same frequency-shift grid and detector intensity scale
as the measured CTS spectrum. The researcher explicitly selects the number
of retained principal components; the code does not choose a component count
or fabricate reference spectra.

The reference mean and selected PCA modes define a fixed detector-space
background nuisance subspace. At every nonlinear plasma-parameter trial,
linear least squares profiles the CTS amplitude and the background-mode
coefficients jointly. The measured CTS spectrum is not baseline-subtracted
or otherwise modified. The supplied CTS frequency-response operator is
applied only to the physical CTS forward spectrum; the reference-derived
background remains in detector space because it is already measured there.

The result returns background_model, background_spectrum,
background_coefficients, background_coefficient_names,
background_pca_components_used, the conditional
background_coefficient_covariance, the propagated detector-space
background_spectrum_covariance, and its per-frequency 1σ uncertainty.
When spectral uncertainty is supplied, the linear nuisance covariance uses
that weighting; otherwise it is scaled by the fitted residual variance.
Without reference spectra, the existing single constant-offset nuisance
model remains unchanged. This option is electrostatic-only; passing it to
the Full-EM dispatcher is not supported.

This is a reference-spectrum PCA nuisance model, not a claim to reproduce
the full physics-informed PCA method of Verdier et al. It is motivated by
the demonstrated importance of structured, time-varying CTS backgrounds:
T. Verdier et al., “Robust monitoring of thermal and fast ions using
collective Thomson scattering: Combining physics- and data-driven
background estimation,” Review of Scientific Instruments 96 (2025),
043513, DOI: 10.1063/5.0238948. Their physics-informed PCA method used
on-pulse modulation and diagnostic-specific temporal structure; the TIS
implementation uses only the calibrated reference spectra explicitly
provided by the researcher and does not infer missing modulation states.

All reference spectra are treated as prepared/calibrated measurements.
No instrument calibration, detector correction, empirical validation or
experimental validation is performed or claimed. Uncertainty in the reference ensemble and in the estimated PCA basis itself
is not propagated by this conditional route. The reported covariance is
conditional on the fixed supplied PCA basis; the returned background
coefficients remain nuisance-fit outputs, not independent physical plasma
parameters.

## Full-EM independent model-discrepancy covariance

The Full-EM measured-spectrum inverse now accepts an optional
`model_discrepancy_covariance` on the calibrated frequency-shift grid. It is
an independent additive model-error covariance, not a fitted background and
not an estimate derived from the observed residual.

The solver combines it with exactly one supplied measurement-uncertainty
representation:

- `spectral_covariance`: full measured frequency-bin covariance; or
- `signal_uncertainty`: pointwise 1-sigma measurement uncertainty, converted
  to a diagonal covariance.

The total covariance used for GLS is
`C_total = C_measurement + C_model_discrepancy`. If no measurement
uncertainty is supplied, the declared discrepancy covariance alone is used
and the result explicitly reports that no measurement-noise covariance was
provided. If neither source is supplied, the fit remains unweighted and its
local covariance is residual-variance-scaled.

The discrepancy covariance must be finite, symmetric and positive
semidefinite. The total covariance must be positive definite for Cholesky
whitening. The solver does not add diagonal jitter, clip eigenvalues, infer
missing covariance, or silently regularize a singular covariance. The result
returns the supplied discrepancy covariance, effective covariance, weighting
status and covariance interpretation.

This is a conditional additive covariance model. It does not propagate
cross-covariance between measurement noise and model discrepancy, does not
infer discrepancy hyperparameters, and does not cover omitted uncertainty in
ray/equilibrium, beam geometry, plasma composition or polarization unless
those are supplied through their separate supported covariance contracts.


## Path-resolved physical CTS background integration (Full-EM)

`analyze_full_em_cts_spectrum` accepts either the existing
`physical_background_spectrum_W_Hz` (already expressed on the measured
receiver grid) or the new `physical_background_radiative_transfer` mapping,
but never both. The mapping requires path-resolved electron temperature and
density, ion densities and charge states, path lengths, an explicit Gaunt
factor, incident boundary radiance, receiver etendue and end-to-end
efficiency. It also requires the
`ece_absorption_coefficient_m_inverse` key; an explicit `None` means ECE is
intentionally omitted rather than silently assumed absent. ECE absorption may
be a single-effective-mode array with shape `(n_path,n_frequency)`, or a
mode-resolved array with shape `(n_path,n_frequency,n_mode)`. The latter
requires `ece_mode_weights`, a measured non-negative receiver power-weight
vector of length `n_mode` summing to one. Each mode is propagated separately
with its own absorption coefficient before receiver-weighted radiances are
combined. This extension does not model coherent cross-mode coupling or mode
conversion.

The absolute receiver frequency is formed as
`probe_frequency_Hz + frequency_shift_Hz`. Free-free and LTE mode-resolved
ECE transfer is integrated along the supplied path, converted from radiance
to receiver spectral power, and passed through the measured frequency-response
operator exactly once. The resulting background is included additively in
the measured-grid fit. TIS does not estimate any plasma profile, ECE opacity,
Gaunt factor, etendue, efficiency, geometry or calibration.

An optional positive-semidefinite
`physical_background_covariance_W2_per_Hz2` is defined on the final measured
frequency grid. It is added to the independently supplied model-discrepancy
covariance for GLS. This addition assumes the two covariance sources are
independent; their cross-covariance is not inferred. If no uncertainty is
supplied, the background remains fixed and conditional.

The free-free kernel is non-relativistic and unmagnetized. ECE absorption
must come from a separate magnetized kinetic/radiative-transfer calculation.
Independent mode transfer is supported when mode-resolved opacity and measured
receiver-mode weights are supplied; coherent mode conversion and cross-mode
coupling remain outside this model. The implementation does not approximate
ECE opacity from scalar magnetic-field magnitude. It also does not add a
collision operator to the CTS dielectric response: numerical resonance
damping remains distinct from physical collisional kinetics. No experimental
validation is claimed; tests are synthetic software-contract checks only.

### Collisional longitudinal response kernel (TS-01 groundwork)

`tokamak_analysis.cts_collisional_bgk.conserving_bgk_longitudinal_susceptibility`
implements a multispecies, one-dimensional, isothermal BGK *density-response
matrix*. The linearized relaxation target preserves each species' particle
number and chooses a shared perturbed flow so that the sum of species momentum
moments of the collision term vanishes. The routine returns
`delta(n_i)/delta(U_j)` for externally applied species potential energies,
with explicit collision frequencies and Gauss-Hermite quadrature.

This is deliberately **not** advertised as a completed collisional CTS spectrum.
The kernel is BGK rather than the velocity-dependent linearized Landau
operator; it assumes a common equilibrium temperature and flow, does not
conserve total energy, and does not yet provide the equilibrium fluctuation
source / fluctuation-dissipation closure needed to construct the partial
electron-density dynamic structure factor measured by CTS. It is not connected
to the electrostatic or Full-EM spectral-density dispatchers. In particular,
its response matrix must not be treated as a measured CTS power spectrum or
used as an arbitrary resonance-width correction. The Full-EM magnetized
gyroharmonic collision operator remains unimplemented. TS-01 therefore
remains open pending a validated fluctuation closure and spectral integration.


## Dashboard input for Full-EM receiver backgrounds

The Full-EM CTS dashboard exposes the same physical receiver-background
contract as the Python inversion API. An optional JSON mapping may supply
path-resolved electron temperature and density, ion densities and charge
states, a scalar/frequency/path/path-frequency Gaunt factor, ECE absorption
coefficients (including independently propagated mode-resolved opacity and
receiver mode weights), incident radiance, receiver etendue, and end-to-end
efficiency. The mapping is passed unchanged to
`physical_background_radiative_transfer`; TIS does not infer path profiles,
Gaunt factors, ECE opacity, mode weights, etendue, or efficiency. Set the ECE
absorption field to JSON `null` only when ECE is intentionally omitted.
An optional frequency-bin covariance CSV is passed as
`physical_background_covariance_W2_per_Hz2` and is included in the
conditional GLS covariance after validation. The measured frequency-response
operator is applied once by the physical-background builder.

The dashboard controls do not turn the free-free approximation into a
magnetized/relativistic emissivity model. The free-free emissivity remains
non-relativistic Maxwellian with an explicit Gaunt factor; ECE opacity remains
an externally supplied mode-resolved kinetic/radiative-transfer result. The
independently propagated mode approximation neglects coherent mode coupling.


The Full-EM dashboard also accepts an optional JSON array of at least 20
precomputed, independently ray-traced geometry realizations. Each array item
must be a complete geometry object containing
`frequency_resolved_geometry`, including per-frequency scattering samples,
beam-overlap and path-length weights. The dashboard samples this ensemble and
reruns the full nonlinear spectral inversion for each draw through
`propagate_full_em_cts_ray_geometry_uncertainty`; it does not perturb fitted
parameters after the fact. The ensemble must have been generated by resampling
the equilibrium and beam inputs and recomputing incident/receiver rays and
scattering-volume overlap for each realization. It is not sufficient to
duplicate or randomly perturb one nominal geometry. A separate local external
covariance upload cannot be combined with this ensemble because that would
discard cross-covariance; correlated inputs must be represented in the same
joint ensemble. The resulting covariance is conditional on the supplied
ensemble and remains distinct from experimental validation.


## Drifted relativistic Maxwell-Juttner electron populations

Full-EM CTS now accepts an explicit
`drifted_relativistic_maxwell_juttner` electron distribution with
`temperature_eV` and signed parallel `drift_velocity_m_s`. The momentum-space
equilibrium is evaluated by a Lorentz boost of the isotropic rest-frame
Maxwell-Juttner distribution. The parallel momentum quadrature is widened for
the boosted support, and the momentum derivatives are evaluated by the chain
rule before the relativistic kinetic response and three-wave vertex are
assembled. The distribution is normalized on the numerical momentum grid.

Do not also supply a non-zero `electron_flow_projection_m_s`: drift is encoded
in the distribution itself, and a second frequency-shift approximation would
double count it. The drift must be strictly subluminal. Quadrature convergence
must be checked for the selected temperature and drift; this is a numerical
forward-model capability, not an assertion that a drifted VDF has been
identified from a single CTS spectrum.


## Spatial Full-EM covariance and measured-response contract

The spatial-profile inverse accepts exactly one measurement-noise representation
(`spectral_covariance` or `signal_uncertainty`) and an optional independent
`model_discrepancy_covariance`. The effective GLS covariance is the sum of
the measurement covariance and the supplied discrepancy covariance, added
exactly once. The discrepancy matrix must be finite, symmetric and positive
semidefinite; the effective covariance must be positive definite for
Cholesky whitening. No diagonal jitter, eigenvalue clipping or residual-derived
discrepancy is applied. Results retain the measurement covariance, discrepancy
covariance, effective covariance and an explicit weighting status separately.
Regularization remains a deterministic prior penalty and is never added to the
measurement covariance.

The spatial inverse now enforces the same calibrated measured-response
contract as the Electrostatic and ordinary Full-EM CTS paths: the response
matrix must be finite, non-negative, square and already have unit-sum columns.
It is applied unchanged after scattering-volume integration; TIS does not
silently renormalize the supplied base operator. These are software contract
and synthetic regression checks, not instrument calibration or experimental
validation.

## Full-EM physical-background zero-state output

The measured-spectrum inversion preserves the lower-level radiative-transfer
background state in its top-level result:

- `physical_background_zero` is `true` only for an explicitly modelled
  zero-radiance background or an explicitly supplied receiver-grid spectrum
  whose bins are all exactly zero;
- it is `false` for a nonzero supplied background, including incident
  radiance with no plasma emission/absorption;
- it is `null` when no physical background was supplied. An omitted input is
  unknown and is not silently interpreted as a measured zero background;
- `physical_background_zero_status` preserves the radiative-transfer
  distinction or identifies the fixed-spectrum/omitted-input state.

The additive fitted spectrum still contains a numerical zero vector when no
background is supplied, because no additive component is available to add.
That numerical placeholder is not a physical assertion that the receiver
background is zero. Regression tests cover the zero, nonzero, incident-only,
omitted and malformed-status cases.

