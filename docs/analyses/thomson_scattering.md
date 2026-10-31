# Thomson Scattering — Analysis 1

## Primary contract

Analysis 1 starts from measured, wavelength-calibrated and intensity-calibrated optical Thomson spectra, not from electron-density or electron-temperature profiles. Its primary input is a complete spectrum for every spatial channel, the supplied spatial coordinate, and the measured instrument response. Detector calibration, raw-count correction and instrument calibration are outside this analysis.

The execution path is measured calibrated spectrum → explicitly selected physical forward model → measured-response convolution → physical parameter inference → channel-resolved diagnostics → optional spatial-profile reconstruction. No pre-derived \(n_e\) or \(T_e\) profile is accepted by the canonical analysis.

## Selected physical models

The model must be selected explicitly in both the dashboard and the public Python API. The current optical Analysis 1 shortlist contains exactly three models:

1. **Non-collective Maxwellian.** The standard incoherent single-particle Thomson spectral model for an isotropic non-relativistic Maxwellian electron distribution. The measured response is convolved with the calculated spectrum. The fitted centroid yields the electron-flow projection along the scattering vector. The classical model rejects inputs outside its declared non-relativistic temperature/flow regime.
2. **Non-collective Kappa.** A finite-variance isotropic Kappa electron distribution represented by its one-dimensional Student-t projection. The analysis infers effective electron temperature and the Kappa index. This is a parameterized non-Maxwellian model, not an unrestricted multidimensional velocity-distribution reconstruction.
3. **Non-collective relativistic Maxwell-Juttner.** An isotropic Maxwell-Juttner rest-frame electron distribution, Lorentz-boosted along the scattering vector. The forward model evaluates the exact elastic-Thomson photon-frequency ratio and Lorentz aberration, and transforms the supplied incident polarization into each electron rest frame before evaluating the polarized differential cross-section. It is the selected option for high-temperature regimes where the classical non-relativistic approximation is not appropriate. Photon recoil/Compton corrections are neglected; the model is restricted to the Thomson limit and does not infer arbitrary anisotropic relativistic distributions.

Optical collective dielectric models are not part of this three-model Analysis 1 shortlist. Microwave collective Thomson scattering (CTS) is a separate registered subanalysis with its own explicitly selected electrostatic and magnetized full-electromagnetic forward models. No physical model is switched automatically.

## Input contract

| Input | Unit / shape | Requirement |
|---|---|---|
| wavelength_nm | nm; shared wavelength axis or channel × wavelength | Required calibrated wavelength axis |
| spectra | calibrated spectral intensity; channel × wavelength | Required measured calibrated spectra |
| radius_m | m; one per channel | Required supplied spatial coordinate |
| laser_wavelength_nm | nm | Required |
| scattering_angle_rad | rad | Required |
| instrument_response | measured kernel/matrix | Required measured response; identity is accepted only when explicitly justified by already response-corrected spectra |
| signal_uncertainty | same unit/shape as spectra | Optional pointwise 1σ uncertainty |
| joint_spectral_covariance | channel-major full covariance | Optional; joint GLS is supported for non-collective Maxwellian and Kappa fits |
| incident_direction / scattered_direction | dimensionless 3-vectors | Required for relativistic Maxwell-Juttner; also required for absolute-density inference |
| incident_polarization | transverse dimensionless 3-vector | Required for relativistic Maxwell-Juttner; also used by absolute-density calculation |
| absolute-density optical inputs | W, m, sr, efficiency and polarization geometry | Required for absolute density from calibrated W/nm spectra |

Input spectra are assumed to be prepared and calibrated. The analysis does not perform wavelength calibration, detector calibration, raw-count correction, background correction of the measurement system, or instrument-response measurement. A local background nuisance parameter in the physical spectral fit is part of inference, not detector calibration.

## Instrument response and spectral-distribution diagnostics

The supplied measured response is applied in the physical forward model. Spectral-density samples are converted to bin-integrated power before applying the response and divided by measured-bin widths afterward, supporting nonuniform calibrated wavelength grids. No Gaussian line-spread function is imposed by the primary physical fit.

For the three optical models, supplied response-matrix uncertainty is propagated by Monte Carlo perturbation and refitting where supported by the model-specific forward path. The spectral-distribution subanalysis is independently registered as entry 81 and can run directly from one calibrated measured spectrum even when the primary physical fit fails. It reports signed wavelength-space centroid, observed variance/width, skewness and excess kurtosis, together with a response-convolved Gaussian shape diagnostic. It does not claim to reconstruct an arbitrary electron velocity distribution. If moments are not identifiable for a channel, the subanalysis reports that channel as unavailable without aborting the primary physical inference.

## Electron density and identifiability

A normalized non-collective spectral shape alone does not determine absolute electron density. Absolute density is calculated only when the spectral intensity is explicitly declared calibrated in W/nm and all required optical factors are supplied: laser power at the scattering volume, interaction length, collection solid angle, optical efficiency, incident direction, incident polarization and scattered-light direction. If the contract is incomplete, density is reported unavailable rather than fabricated from fitted amplitude.

The physical fit must pass its identifiability and fit-quality gates. Rank-deficient, ill-conditioned or unacceptable fits are rejected rather than returning unsupported physical parameters. No regime or model is silently inferred from missing metadata.

## Outputs

- Channel-resolved electron temperature and uncertainty.
- Electron-flow projection along the scattering vector and its uncertainty; no unsupported 3-D velocity reconstruction.
- Absolute electron density only where calibrated W/nm intensity and complete optical geometry/polarization make it identifiable.
- Kappa index and uncertainty for the non-collective Kappa model.
- Fitted spectra, residuals, local fit covariance, identifiability diagnostics and model provenance.
- Model-independent spectral-distribution diagnostics for each channel where the signed moments are identifiable.
- Spatial PCHIP and/or generalized-cross-validation spline reconstructions, with propagated uncertainty bands where the supplied covariance supports them.

## Uncertainty and deliberate limitations

- Full channel-major generalized least squares with Cholesky whitening is implemented for non-collective Maxwellian and Kappa fits. The relativistic Maxwell-Juttner fit currently uses channel-local fit covariance and response-matrix Monte Carlo; full cross-channel spectral GLS is not implemented for that model.
- Laser-wavelength and scattering-angle uncertainty propagation is supported for the classical Maxwellian/Kappa fits. Those uncertainty inputs are rejected for the relativistic Maxwell-Juttner model rather than silently ignored. Incident-polarization and full beam-direction covariance are not propagated for the relativistic model.
- Absolute-density uncertainty uses either the supplied complete relative covariance of density-inversion inputs or the explicit scalar uncertainty terms. Unprovided cross-correlations and model discrepancy are not silently set to measured values.
- Kappa is a parameterized finite-variance distribution, not an unrestricted VDF inversion.
- The relativistic Maxwell-Juttner model assumes an isotropic rest-frame distribution and bulk flow parallel to the scattering vector; photon recoil and Compton corrections are neglected.
- Optical collective dielectric inference is outside this Analysis 1 shortlist. Microwave CTS is covered by its separate registered workflow.
- An integrated spectral integral is reported in watts only when the calibrated spectral-density unit is explicitly W/nm.
- No independent experimental validation is claimed. Automated tests are software/numerical consistency tests only.

## Spatial profile reconstruction

The registered “Thomson spatial profile reconstruction” subanalysis starts from measured calibrated multi-channel spectra, runs the canonical physical inference, and reconstructs every identifiable radial parameter profile using shape-preserving PCHIP and/or a generalized-cross-validation smoothing spline. It never accepts pre-derived \(n_e/T_e\) profile points and never extrapolates beyond the measured radial interval. The Kappa-index profile is included only for the non-collective Kappa model.

The reconstruction propagates the available per-channel fit covariance and any supplied cross-channel spectral or absolute-density input covariance. A profile is reported only where its source physical parameter is identifiable; no missing channel values are fabricated or extrapolated.

No public Analysis 1 or subanalysis entry point accepts already-derived electron-density or electron-temperature profile points. Spatial reconstruction always starts from measured calibrated spectra and invokes the canonical physical forward-inference engine.

## Scientific basis and recent literature

The shortlist is intentionally limited to three commonly useful optical forward models, with recent peer-reviewed literature informing the selected scope:

1. M. Carole et al., “A photonic model of the High-Resolution Thomson Scattering diagnostic of WEST tokamak,” *Journal of Instrumentation* **20** (2025), P07014, DOI: [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014) — optical Thomson forward modelling and parameter inference.
2. C. R. Skolar, W. J. Longley and L. V. Goodwin, “Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering,” *Physics of Plasmas* **32** (2025), 043907, DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966) — non-Maxwellian Kappa spectral response.
3. K. Sakai, K. Tomita, T. Hoshi and R. Yasuhara, “Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions,” *Journal of Instrumentation* **21** (2026), C01002, DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002) — relativistic Maxwellian and Kappa spectra in high-temperature plasmas.
4. T. Kawate and M. Goto, “Spectroscopic Diagnostics of Non-Maxwellian Electron Velocity Distribution Function in the Large Helical Device,” *Journal of Fusion Energy* **44** (2025), 44, DOI: [10.1007/s10894-025-00516-0](https://doi.org/10.1007/s10894-025-00516-0) — non-Maxwellian electron-distribution context in a magnetic-confinement plasma; Thomson measurements provide electron-temperature/density reference values.
5. M. Amarika et al., “ITER Core Plasma Thomson Scattering diagnostic design,” *Fusion Engineering and Design* **203** (2024), 114416, DOI: [10.1016/j.fusengdes.2024.114416](https://doi.org/10.1016/j.fusengdes.2024.114416) — ITER core electron-temperature/density measurement requirements over 0.5–40 keV.

These references motivate the selected physical scope; TIS does not claim to reproduce every method or result in those papers.

## Dashboard access

The Spectral Analysis workspace exposes the canonical multi-channel optical Thomson inference. The Advanced Analysis profile workflow routes to the same canonical engine. The microwave CTS workflow separately exposes electrostatic and magnetized full-electromagnetic forward models.


## Independent spectral-distribution subanalysis (registry entry 81)

The callable `tokamak_analysis.thomson_distribution.analyze_thomson_spectral_distribution` is registered independently and accepts a single measured calibrated spectrum. It does not require a successful primary Maxwellian/Kappa/Maxwell-Juttner fit. The background must be explicitly supplied as `background_signal`, `background_mask`, or a researcher-selected `background_fraction`; the code does not silently designate the first/last 10% of the wavelength range as background. A measured response matrix is used directly. The legacy `thomson_spectrum.fit_thomson_spectrum` adapter also requires a measured response matrix and no longer synthesizes a Gaussian response from a scalar width.

The Kappa physical fit reports the Student-t probability captured by the measured wavelength window. It does not impose the Gaussian ±3σ coverage rule on a heavy-tailed distribution; if the fitted model has appreciable tails outside the measured window, the extrapolated full-model area is explicitly flagged.

## Literature traceability

The Kappa forward model implemented here is the analytic one-dimensional Student-t projection of an isotropic finite-variance Kappa velocity distribution. Skolar, Longley and Goodwin (2025, DOI 10.1063/5.0261966) develop a generalized plasma-dispersion-function approach for non-Maxwellian Thomson response; that is related literature but is **not** the same numerical algorithm as this Student-t fit. Sakai et al. (2026, DOI 10.1088/1748-0221/21/01/C01002) provide Monte Carlo reference spectra for arbitrary electron distributions, including relativistic Maxwellian and Kappa cases; TIS's deterministic quadrature is not claimed to reproduce their Monte Carlo method. These papers are references for comparison and equation traceability, not evidence of experimental validation of this repository.
