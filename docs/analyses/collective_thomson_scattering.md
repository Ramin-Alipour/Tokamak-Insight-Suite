# Microwave Collective Thomson Scattering (CTS)

## Scope and model selection

The CTS family is separate from optical Thomson Analysis 1. Its public dispatcher is:

- \`tokamak_analysis.cts_analysis.analyze_cts_spectrum_model(model="electrostatic", ...)\`
- \`tokamak_analysis.cts_analysis.analyze_cts_spectrum_model(model="full_electromagnetic", ...)\`

The model must be selected explicitly. The software never silently substitutes the electrostatic model for the magnetized full-EM model or vice versa.

All spectra and frequency axes are assumed to be measured, prepared and calibrated. The CTS workflows do not calibrate the instrument, correct raw detector counts, infer a missing frequency axis, or perform experimental validation.

## Electrostatic longitudinal CTS

The electrostatic path remains an unmagnetized multi-species longitudinal dielectric model. It computes species susceptibilities, the longitudinal dielectric function, the dynamic structure factor and convolution with the supplied measured frequency-response matrix.

It supports the declared Maxwellian background and fast-ion options, including a one-dimensional projected tabulated fast-ion VDF. Ion species, charge states, masses, charge-density fractions and the electron-flow projection are explicit inputs. The electrostatic model's common ion-temperature/projected-ion-flow assumptions remain part of that path; they are not silently transferred to full-EM.

## Magnetized full-electromagnetic CTS

The full-EM path is implemented in \`tokamak_analysis.cts_full_em\`. It combines:

1. Magnetized Maxwellian kinetic conductivity and screened density/current/electric/magnetic fluctuation covariance, retaining all supplied auto- and cross-correlations.
2. The low-temperature Bindslev \`Gamma^{-1}\` / \`X\` source-current closure for Maxwellian bulk-electron three-wave coupling.
3. Species-resolved thermal-ion temperature inference. Each supplied ion species has its own fitted temperature; composition and flow vectors remain fixed external inputs.
4. Optional relativistic gyrotropic electron and fast-ion populations, with relativistic Vlasov linear response, Klimontovich shot-noise covariance and a second-order relativistic Vlasov three-wave current vertex.
5. Optional frequency-resolved cold-plasma ray tracing and path-integrated Gaussian beam-overlap weighting.

### Relativistic kinetic populations and three-wave vertex

Implemented in \`tokamak_analysis.cts_relativistic\`:

- Relativistic Maxwell-Juttner momentum distribution for a thermal relativistic electron population. Its temperature follows the fitted electron temperature when \`electron_distribution={"kind":"relativistic_maxwell_juttner"}\` is supplied without an explicit temperature.
- Relativistic anisotropic bi-Gaussian momentum distributions with explicit perpendicular and parallel temperatures.
- Caller-supplied tabulated gyrotropic distributions \(f(p_\perp,p_\parallel)\).
- Relativistic linear Vlasov response using \(v=p/(\gamma m)\) and \(\Omega/\gamma\).
- Klimontovich particle shot-noise covariance for the supplied relativistic populations.
- A second-order relativistic Vlasov three-wave current vertex, evaluated from the first-order perturbations and momentum derivatives of the supplied VDF.

When an electron distribution is explicitly supplied, the Maxwellian electron response and the Bindslev low-temperature electron nonlinear source are replaced by the relativistic kinetic electron response and second-order vertex. When no electron distribution is supplied, the Maxwellian bulk-electron Bindslev closure remains active. The fitted dimensionless theta_e = T_e/(m_e c²) is reported. A researcher may supply `bindslev_validity_theta_e_limit`; the fit is rejected if theta_e exceeds that explicitly selected asymptotic screening limit. If no limit is supplied, the result is marked `not_quantitatively_screened`; the software does not invent a universal cutoff. This is an explicit model choice, not an automatic relativistic correction to every Maxwellian run.

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

## Full-EM input contract

Required inputs include:

- Calibrated frequency-shift axis and calibrated CTS spectral power.
- Probe frequency and measured instrument-response matrix.
- Researcher-supplied initial electron density and temperature estimates.
- Explicit ion species metadata, charge-density fractions and one initial temperature per ion species.
- Optional researcher-supplied parameter bounds keyed by electron density, electron temperature and species-specific ion temperatures. If omitted, the positive log-parameters are not restricted by default scientific ranges.
- Magnetic-field vector.
- Explicit electron-flow vector and one ion-flow vector per ion species.
- Incident/scattered wave directions and polarization vectors.
- Measured spectral uncertainty or a full frequency-bin covariance matrix.

Optional inputs include:

- Relativistic electron distribution metadata.
- Relativistic fast-ion population metadata.
- Positive-definite spectral covariance.
- Frequency-resolved ray-traced scattering-volume geometry.
- Explicit harmonic order, quadrature order and resonance damping.

No zero flow, ion composition, polarization, VDF or equilibrium profile is silently assumed. A zero vector is an explicit stationary-flow input.

## Inversion and uncertainty

The full-EM inversion uses least-squares in logarithmic positive parameters, profiles spectral amplitude and constant background as nuisance parameters, and fits electron density, electron temperature and one temperature per supplied thermal-ion species. No scientific parameter bounds are imposed by default; optional bounds must be supplied explicitly by the researcher. Flow vectors, species composition and supplied non-Maxwellian VDF shapes remain external inputs. The relativistic Maxwell-Juttner electron distribution follows the fitted electron temperature when no separate temperature is provided.

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

The public contract distinguishes Electrostatic and Full-EM inputs. The Electrostatic path alone consumes the scalar electron-flow projection, common projected ion-flow initial estimate, Electrostatic inversion-method/global-search controls, Electrostatic response-matrix Monte Carlo controls and its supported external-parameter uncertainty mapping. The Full-EM path alone consumes the magnetic-field vector, electron/ion flow vectors, incident/scattered directions and complex polarizations, harmonic/quadrature orders, optional relativistic distributions, frequency-resolved ray geometry and scattering-volume sample controls. Density/temperature initial estimates, ion species/composition, measured frequency axis/spectrum, probe frequency and measured response are shared. These conditions are encoded in `DIAGNOSTIC_REGISTRY.json` using `required_when_model`; a single flat list of mandatory inputs must not be used for both models.

When `damping_rate_rad_s` is omitted, Full-EM uses the documented numerical regularization

\[
\gamma = \max(2\pi f_0\times 10^{-7},\;1\;\mathrm{rad\,s^{-1}}).
\]

This is not a physical collision frequency. The resolved value and whether it was caller-supplied or defaulted are returned in `numerical_controls`. Harmonic order, quadrature order and damping should be varied in numerical convergence studies before interpreting the result. Ray-tracing APIs require an explicit `mode_index`; no propagation branch is silently selected.

If no pointwise uncertainty or spectral covariance is supplied to Electrostatic CTS, the fit is unweighted. The implementation does not invent a noise scale from the measured spectrum; the result reports `spectral_weighting_status=unweighted_no_measurement_uncertainty_supplied`, and reduced chi-square is unavailable.
