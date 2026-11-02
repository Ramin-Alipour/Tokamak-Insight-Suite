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

It supports the declared Maxwellian background and fast-ion options, including a one-dimensional projected tabulated fast-ion VDF. Ion species, charge states, masses and charge-density fractions are explicit external inputs. The inversion fits electron-flow and common projected ion-flow components from the measured CTS spectrum. Density and temperature optimizer seeds are optional: omitted values are initialized automatically from measured spectral shape and width, while the final parameters are determined by the full forward-model fit. `electron_velocity_projection_m_s` and `ion_velocity_initial_m_s` are optional numerical optimizer seeds; omission uses zero as a numerical starting value only. Neither is a fixed measured plasma parameter or a stationary-flow assumption. The dashboard labels the electron-flow value as an optional seed and no longer requires the user to enter zero to assert stationary flow.
The electrostatic dashboard also accepts an optional joint external-parameter covariance CSV with an explicit ordered parameter list. Supported names are `probe_frequency_Hz`, `scattering_angle_rad`, and `ion_fraction_i` (zero-based species index). This covariance is mutually exclusive with the separate scalar uncertainty fields; the uploaded matrix is propagated through correlated Monte Carlo refits. The covariance is user-supplied and is not inferred from the spectrum.

## Magnetized full-electromagnetic CTS

The full-EM path is implemented in \`tokamak_analysis.cts_full_em\`. It combines:

1. Magnetized Maxwellian kinetic conductivity and screened density/current/electric/magnetic fluctuation covariance, retaining all supplied auto- and cross-correlations.
2. The low-temperature Bindslev \`Gamma^{-1}\` / \`X\` source-current closure for Maxwellian bulk-electron three-wave coupling.
3. Species-resolved thermal-ion inference. Each supplied ion species has its own fitted temperature and flow projection along the scattering wavevector; ion composition remains a fixed external input.
4. Optional relativistic gyrotropic electron and fast-ion populations are supported by the forward spectral-density function. The current measured-spectrum inversion rejects non-null `electron_distribution`; it fits the Maxwellian bulk-electron branch because a drifted relativistic electron VDF fit is not implemented. The dashboard therefore does not offer the forward-only VDF input as an inversion control. Explicit relativistic gyrotropic electron distributions remain available to the forward spectral-density API, not to measured-spectrum inversion. Fast-ion population shapes supplied to Full-EM inversion likewise remain fixed external inputs. Fast-ion populations, when supplied to inversion, remain fixed external distributions rather than fitted VDFs.
5. Optional frequency-resolved cold-plasma ray tracing and path-integrated Gaussian beam-overlap weighting.

### Relativistic kinetic populations and three-wave vertex

Implemented in \`tokamak_analysis.cts_relativistic\`:

- Relativistic Maxwell-Juttner momentum distribution for a thermal relativistic electron population. Its temperature follows the fitted electron temperature when \`electron_distribution={"kind":"relativistic_maxwell_juttner"}\` is supplied without an explicit temperature.
- Relativistic anisotropic bi-Gaussian momentum distributions with explicit perpendicular and parallel temperatures.
- Caller-supplied tabulated gyrotropic distributions \(f(p_\perp,p_\parallel)\).
- Relativistic linear Vlasov response using \(v=p/(\gamma m)\) and \(\Omega/\gamma\).
- Klimontovich particle shot-noise covariance for the supplied relativistic populations.
- A second-order relativistic Vlasov three-wave current vertex, evaluated from the first-order perturbations and momentum derivatives of the supplied VDF.

When an electron distribution is explicitly supplied to the forward model, the Maxwellian electron response and the Bindslev low-temperature electron nonlinear source are replaced by the relativistic kinetic electron response and second-order vertex. When no electron distribution is supplied, the Maxwellian bulk-electron Bindslev closure remains active. The dimensionless theta_e = T_e/(m_e c²) is reported. In measured-spectrum inversion, `bindslev_validity_theta_e_limit` applies to the Maxwellian Bindslev closure; the inversion rejects explicit `electron_distribution` input rather than claiming a relativistic-electron fit. If no screening limit is supplied, the Maxwellian result is marked `not_quantitatively_screened`; the software does not invent a universal cutoff.

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

- No relativistic electron-distribution input is accepted by the current measured-spectrum inversion; explicit relativistic electron distributions are forward-model-only until a drifted relativistic VDF inversion is implemented.
- Relativistic fast-ion population metadata.
- Positive-definite spectral covariance.
- Frequency-resolved ray-traced scattering-volume geometry.
- Explicit harmonic order, quadrature order and resonance damping.

Flow projections omitted from the inversion's optional seed inputs are initialized numerically at zero and then fitted; zero is not held fixed as a stationary-flow assumption. Ion composition, polarization, VDF shape and equilibrium profile are not silently inferred and must be supplied where required.

## Numerical convergence and uncertainty scope

The measured-spectrum inversion automatically calls `tokamak_analysis.cts_full_em.assess_full_em_cts_numerical_convergence` after fitting, unless the researcher explicitly sets `numerical_convergence=False`. It evaluates paired increasing `harmonic_order` and `quadrature_order` values while holding fitted physical parameters, external inputs and damping prescription fixed. It reports successive relative L2 and L-infinity spectral changes and a tolerance-based convergence status. The helper can also be called independently for a forward-model convergence study. This is a numerical discretization check only; it is not experimental validation or a physical-model uncertainty estimate.

Pointwise spectral uncertainty or a full frequency-bin covariance is optional for Full-EM inversion. If supplied, it weights the fit; if omitted, the fit is unweighted and the local Jacobian covariance is scaled by residual variance, explicitly not measurement-noise-calibrated. The result preserves this as `conditional_fit_covariance`. When measured response-matrix uncertainty is also supplied, its successful Monte Carlo refit covariance is exposed separately as `instrument_response_parameter_covariance` and in `external_uncertainty_covariance_components`; `fit_covariance` is then the sum of those two components. The combined scope explicitly excludes cross-covariances with other external inputs. The standalone local-input covariance sampler independently validates that its covariance is finite, real, symmetric, positive semidefinite and dimensionally consistent with the ordered base vector before drawing reproducible perturbations; it does not infer or repair an invalid covariance. It does not constitute a complete uncertainty budget for magnetic field, ion composition, beam directions, polarization, ray tracing, beam covariance propagation or supplied kinetic-distribution shape. Those fixed inputs and exclusions are exposed in the result provenance/scope fields. The damping default is a numerical resonance regularization, not a physical collision frequency.

## Inversion and uncertainty

The full-EM inversion uses least-squares in logarithmic positive parameters, profiles spectral amplitude and constant background as nuisance parameters, and fits electron density, electron temperature, the electron-flow projection, and one temperature plus one flow projection per supplied thermal-ion species. The optional flow-projection inputs are optimizer seeds, not fixed flow vectors. No scientific parameter bounds are imposed by default; optional bounds must be supplied explicitly by the researcher. Species composition and supplied fast-ion VDF shapes remain external inputs. Although the forward spectral-density function supports explicit relativistic gyrotropic electron distributions, the current measured-spectrum inversion rejects `electron_distribution` because a drifted relativistic VDF is not implemented; it does not silently assume zero flow or claim a relativistic-electron inversion.

The local Jacobian covariance is reported with fit residuals, identifiability rank and condition number. Full spectral covariance is used for whitening when supplied. The full-EM path does not claim complete propagation of every uncertainty in the supplied equilibrium, ray geometry, beam covariance, VDF or numerical truncation. These are declared model/input uncertainties, not set to zero.

The electrostatic path retains its own differential-evolution, improved-genetic and regularized tabulated-VDF options. Those strategies are not silently applied to the full-EM solver.

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
flags and the numerical power-conservation residual. It rejects missing or
misaligned path/propagation-constant inputs rather than inventing them.

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
