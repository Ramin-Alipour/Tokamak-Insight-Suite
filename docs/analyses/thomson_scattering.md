# Thomson Scattering — Analysis 1

## Primary contract

**Analysis 1 starts from measured, wavelength-calibrated and intensity-calibrated Thomson spectra, not from electron-density or electron-temperature profiles.** Its primary input is a complete spectrum for every spatial channel, with the channel's supplied spatial coordinate and the instrument response specified as a measured response matrix/function. Detector calibration and correction of raw detector counts are outside this analysis.

The primary execution path is:

1. Read measured spectra as a channel-by-wavelength array; the dashboard accepts a long-form table with one row per channel/wavelength sample.
2. Evaluate the selected physical Thomson forward model on each supplied calibrated wavelength grid.
3. Convolve the physical spectrum with the supplied measured instrument line-spread response.
4. Fit each spatial channel and report channel-resolved physical parameters and fit diagnostics.
5. Return all parameters identifiable under the selected physical model, including electron temperature, supported flow projection, and electron density only when the absolute-power/collective identifiability conditions are met.
6. Attach per-channel spectral-distribution diagnostics using the fitted background nuisance component.
7. For the registered spatial-profile subanalysis, reconstruct the inferred channel profiles with PCHIP and/or a GCV smoothing spline, with no extrapolation.

The Spectral Analysis workspace exposes this primary multi-channel spectral-inference workflow. Advanced Analysis routes to the same canonical UI rather than maintaining a second inference implementation. The legacy single-channel API in thomson_spectrum.py is now a compatibility adapter to the canonical engine. It preserves source row order: it does not sort, repair, calibrate, or silently discard samples.

## Input contract

| Input | Unit / shape | Requirement |
|---|---|---|
| wavelength_nm | nm; (wavelength,) or (channel,wavelength) | Required calibrated wavelength axis |
| spectra | calibrated intensity; (channel,wavelength) | Required measured full Thomson spectra |
| radius_m | m; (channel,) | Required supplied coordinate for each channel |
| laser_wavelength_nm | nm | Required |
| scattering_angle_rad | rad | Required |
| instrument_response | callable, 1-D kernel, 2-D matrix or channel-specific 3-D matrix | Required measured response; use an explicit identity matrix only when a delta-like response is justified |
| signal_uncertainty | same unit/shape as spectra | Optional independent pointwise 1-sigma spectral uncertainty |
| joint_spectral_covariance | (channels × wavelengths)² | Optional full positive-definite channel-major spectral covariance; the within-channel marginal covariance is also propagated into spectral-distribution moments |
| instrument_response_uncertainty | same shape as response | Optional 1-sigma response-matrix uncertainty |
| ion_species | explicit species list | Required for collective Maxwellian mode |
| ion_fractions | dimensionless; one per species | Required for multiple ion species; fractions must sum to one |
| ion_temperature_eV | eV, one value per species | Required for the current collective backend; supply from an independent diagnostic when available |
| ion_velocity_m_s | m/s, one 3-vector per species | Required in collective mode; measured vectors or explicit zero vectors declaring stationary ions |
| electron_velocity_m_s | m/s, 3-vector | Required in collective mode; measured vector or explicit zero vector declaring stationary electrons |
| incident_direction / scattered_direction | dimensionless 3-vectors | Required in collective mode; both are expressed in the same coordinate frame as the supplied ion/electron flow vectors and must agree with the scattering angle |
| Absolute-density geometry | W, m, sr, dimensionless efficiency, incident direction, polarization and scattered direction vectors | Required with spectral intensity in W/nm to infer absolute density in the non-collective mode |
| Laser/angle uncertainty | nm / rad | Optional; missing uncertainty inputs are explicitly flagged |
| Optical-factor uncertainty | W, m, sr, dimensionless | Optional; only supplied uncertainty terms are propagated |
| density_relative_covariance | 6×6 relative covariance | Optional within-channel covariance for scattered power and the five multiplicative optical inputs; does not imply cross-channel correlations and is mutually exclusive with scalar uncertainty inputs and joint_spectral_covariance |
| joint_density_relative_covariance | (6N)×(6N) relative covariance | Optional non-collective channel-major full covariance for six density inputs per channel, including cross-channel correlations; returns full density covariance |
| ion_temperature_uncertainty_eV / ion_velocity_uncertainty_m_s / electron_velocity_uncertainty_m_s | eV / m s⁻¹ | Optional collective external-input 1-sigma uncertainties, propagated by Monte Carlo |
| incident_direction_uncertainty / scattered_direction_uncertainty | component-wise 1-sigma 3-vectors | Optional collective beam-direction uncertainties; use instead of scalar scattering_angle_uncertainty_rad, not together |

The measured spectra are assumed to be calibrated intensity data. No wavelength calibration, detector calibration, background correction of the measurement system, or instrument-response measurement is performed by this analysis. The physical fit may include a local background nuisance term, which is part of spectral inference rather than detector calibration.

## Implemented physical model families

### 1. Non-collective Maxwellian

The electron feature is represented by the non-relativistic Maxwellian Doppler spectrum. Its intrinsic width is fitted after applying the supplied instrument-response matrix in the forward model. The fitted spectral centroid yields the electron-flow projection along q = k_scattered − k_incident, not a full three-dimensional velocity vector. With this q convention, the non-relativistic projection is v_q = −c(λ_c − λ_L)/[2 λ_L sin(θ/2)].

Assumptions: non-collective single-particle scattering, isotropic Maxwellian electrons, non-relativistic velocities, and known scattering geometry.

### 2. Non-collective kappa (non-Maxwellian)

A finite-variance isotropic kappa electron velocity distribution is represented by its one-dimensional Student-t projection. The fitted second moment is mapped to an effective electron temperature; the kappa index is also returned. This is a parameterized non-Maxwellian option, not an arbitrary free-form velocity-distribution reconstruction.

Assumptions: non-collective scattering, isotropic kappa distribution, kappa index greater than 1.5, and non-relativistic velocities.

### 3. Collective Maxwellian

The optional collective backend uses the PlasmaPy kinetic dielectric/spectral-density implementation. The model accepts explicit ion species, ion composition fractions, ion temperatures, ion flow vectors and an electron-fluid flow vector. Ion temperatures and flow vectors remain fixed external model inputs: they are not inferred from the Thomson spectrum. Their projected values are calculated from the supplied vectors. If component-wise 1-sigma uncertainties are supplied for ion/electron flow vectors, beam directions, scattering angle, ion temperatures or laser wavelength, the external-input Monte Carlo refits propagate their effects into density, temperature and projected-flow covariance. Flow vectors must be independently measured or explicitly set to zero to declare a stationary-flow assumption. Missing ion temperatures and multi-ion fractions are never silently assumed. PlasmaPy and Astropy are optional dependencies installed with the plasma extra.

The collective model can cover both collective and non-collective scattering regimes through the dielectric spectral-density calculation. The scattering parameter and spectral shape depend on the supplied plasma parameters and geometry.

## Instrument response

The instrument response is represented by a non-negative matrix whose columns are normalized probability distributions for detected wavelength bins conditional on each true wavelength bin. Spectral-density samples are converted to bin-integrated power before applying the response and divided by measured-bin widths afterward, so nonuniform calibrated wavelength grids are handled consistently. The measured response is mandatory and may be supplied as a callable evaluated on the wavelength grid, a 1-D odd-length kernel, a 2-D wavelength-to-wavelength matrix, or a channel-specific 3-D matrix. The same response can be used for all channels, or a separate matrix can be supplied per channel. An identity matrix is accepted only when explicitly supplied and physically justified. This supports asymmetric and wavelength-dependent line-spread functions; a Gaussian response is not imposed.

For the non-collective Maxwellian and kappa fits, supplied response-matrix uncertainty is propagated by Monte Carlo perturbation of the response matrix, followed by refitting. The reported temperature, velocity and (when absolute density is enabled) integrated-power/density uncertainties include the Monte Carlo response contribution when enough valid perturbation fits are obtained. The collective backend propagates supplied response-matrix uncertainty by Monte Carlo refits when a measured 2-D or channel-specific 3-D response matrix is supplied. It also supports Monte Carlo refits for supplied ion-temperature, ion-velocity, electron-velocity and laser-wavelength uncertainties. Each draw refits the full set of spatial channels so cross-channel covariance induced by shared external inputs is retained. Collective geometry uncertainty is now supported: supply scattering_angle_uncertainty_rad for scalar angle uncertainty, or component-wise incident_direction_uncertainty and/or scattered_direction_uncertainty for vector geometry. These two geometry uncertainty routes are mutually exclusive. Beam-vector component perturbations are treated as independent Gaussian uncertainties; cross-component and cross-source correlations are not modeled.

## Electron density and identifiability

A spectral shape alone does not determine absolute electron density in the non-collective regime. Absolute density is calculated only when the spectral intensity is explicitly declared calibrated in W/nm and all required optical factors are supplied:

- incident laser power at the scattering volume;
- interaction length;
- collection solid angle;
- optical efficiency;
- incident laser propagation direction;
- incident polarization vector (perpendicular to the incident beam);
- scattered-light direction (consistent with the specified scattering angle).

The differential Thomson cross-section uses the incident polarization dependence for an unanalysed outgoing polarization measurement. The incident polarization must be perpendicular to the incident beam, and the supplied incident/scattered direction vectors must agree with the specified scattering angle. The fitted Maxwellian and kappa models integrate their analytic intrinsic spectra over the full spectral domain for the power inversion. This avoids treating a finite user-selected wavelength window as the total scattered power. If the absolute-power contract is incomplete, the density profile is reported as unavailable (NaN) with an explicit status; it is not fabricated from the fitted amplitude. Density uncertainty combines the integrated-power uncertainty with the supplied optical-input uncertainty model. A supplied density_relative_covariance is authoritative for the six relative inputs (scattered power, incident power, interaction length, collection solid angle, optical efficiency and polarization/differential-cross-section factor) within each channel; it does not imply cross-channel correlation and must not be combined with separate scalar uncertainty arguments or joint_spectral_covariance. Its scattered-power marginal must include the integrated spectral-area uncertainty. LSF Monte Carlo contributions are added as an independent variance to that marginal when enough valid perturbation fits are obtained. If scattering-angle uncertainty affects the differential cross-section, include that contribution in the sixth-factor covariance because the covariance model replaces separate angle uncertainty in the density inversion. Without that matrix, independent scalar uncertainties are propagated. Each channel lists unavailable uncertainty inputs; missing uncertainties are never silently treated as zero. A supplied joint_density_relative_covariance is ordered channel-major with six factors per channel (non-collective models only) and preserves cross-channel correlations in electron_density_joint_covariance_m6. Collective mode instead exposes its cross-channel density/temperature/flow covariance through collective_external_parameter_covariance and the density-only marginal through electron_density_external_parameter_covariance_m6. This joint matrix is a complete covariance for scattered power and optical factors; do not combine it with density_relative_covariance, joint_spectral_covariance, instrument_response_uncertainty or scalar optical-factor uncertainties. Its scattered-power blocks must already include the spectral-area and response contributions relevant to density.

## Outputs

- Per-channel fitted electron temperature and its uncertainty.
- For non-collective models, the fitted electron-flow projection along the scattering wavevector and its uncertainty. In collective mode, the reported electron and ion flow projections are derived from supplied flow vectors; they are not fitted. When vector/geometry uncertainties are supplied, Monte Carlo estimates their projected-flow uncertainties and returns the full shared-input covariance.
- Per-channel absolute electron density only when identifiable from the selected model and supplied optical inputs.
- Multi-channel radial profiles of the inferred quantities.
- Per-channel fitted spectra, residuals, fit covariance and model assumptions.
- Signed wavelength-space centroid, observed variance/width, skewness, excess kurtosis, positive/signed integrated spectral measure and Gaussian shape-fit diagnostics. The background subtracted for these moments is the fitted nuisance-background component from the physical forward fit.
- When a full channel-major spectral covariance is supplied, its within-channel marginal block is used for spectral-moment and shape-fit uncertainty propagation; independent pointwise errors are not substituted.
- Explicit model, instrument-response usage, input provenance and limitations.
- Per-channel signed spectral centroid, observed width, skewness, excess kurtosis, integrated spectral measure, and response-convolved Gaussian shape diagnostics.
- Parameterized kappa index for the non-collective kappa model.
- Spatial PCHIP/GCV reconstructions, gradients and covariance-based Monte Carlo bands in the registered profile subanalysis.

## Deliberate limitations

- Collective non-Maxwellian kinetic distributions are not implemented; the collective backend is Maxwellian. The non-collective kappa option is parameterized and does not reconstruct an arbitrary VDF.
- The kappa model is a parameterized isotropic non-Maxwellian approximation, not an arbitrary electron VDF inversion. A single projected spectrum does not uniquely determine a general multidimensional electron velocity distribution without additional independent angular/spectral constraints and a specified regularized inversion model.
- Spectral moments and Gaussian shape diagnostics describe the observed wavelength-space distribution; they are not a substitute for the physical Maxwellian/Kappa/collective forward inference and are not interpreted as a unique VDF. Moment uncertainty propagation uses the full within-channel covariance when supplied, or independent pointwise errors when that is the supplied uncertainty model. Uncertainty in the fitted background nuisance parameters is not yet jointly sampled with the spectral moments.
- An integrated spectral integral is reported in watts only when the calibrated spectral-density unit is explicitly W/nm. Otherwise it is returned as integrated_spectral_signal_input_units_nm in the input intensity unit multiplied by nm.
- Absolute density in non-collective mode requires absolute W/nm intensity and complete optical geometry; spectral shape alone is insufficient.
- The collective fit uses a log-parameterized least-squares inversion for positive density and temperature with adaptive bounds around explicit user starting estimates. Ion temperatures and ion/electron flow vectors remain external inputs, not fitted parameters. Supplied ion-temperature, flow-vector, laser-wavelength, scattering-angle/beam-direction and measured response-matrix uncertainties can be propagated by Monte Carlo refitting; the resulting covariance includes per-channel density, temperature, electron-flow projection and species-resolved ion-flow projections. These contributions are conditional on the specified uncertainty distributions and successful perturbation fits.
- Optional joint_spectral_covariance supplies a full channel-major covariance matrix spanning all wavelength bins and channels. It is mutually exclusive with signal_uncertainty: the full covariance is the complete spectral-noise model. A joint generalized least-squares fit uses Cholesky whitening and returns cross-channel parameter covariance for non-collective Maxwellian and Kappa models. Integrated scattered power and absolute density are recomputed from the final joint-fit spectrum, rather than retaining values from the superseded independent fits. Collective and custom models do not yet use this joint GLS path.
- No independent experimental validation is claimed. Automated tests verify software behavior and mathematical consistency only.

## Scientific basis and model selection

The implementation uses three focused physical model families rather than an unbounded list of fitting options:

1. The Maxwellian collective/non-collective spectral-density formulation follows the kinetic dielectric treatment documented by PlasmaPy. Its spectral-density function explicitly supports one or more ion species, ion temperatures and drift velocities.
2. Recent experimental CTS work demonstrates the importance of robust spectral/background treatment for thermal- and fast-ion monitoring (Verdier et al., Review of Scientific Instruments 96, 043513, 2025, DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948)).
3. Recent non-Maxwellian Thomson forward-model work motivates retaining a non-Maxwellian option rather than treating a Gaussian as a universal spectrum (Skolar, Longley & Goodwin, Physics of Plasmas 32, 043907 (2025), DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966)). The kappa implementation here is a bounded parameterized approximation and is not claimed to reproduce that paper's general arbitrary-distribution solver.
4. Recent multi-regime analysis of Thomson spectra reports inference in both collective and non-collective regimes using physical spectral models (Eisenbach et al., Journal of Physics D: Applied Physics 58, 035202, 2025, DOI: [10.1088/1361-6463/ad89d6](https://doi.org/10.1088/1361-6463/ad89d6)).
5. Recent work develops a Monte Carlo forward-spectrum method for arbitrary electron distributions, including relativistic Maxwellian and kappa cases (Sakai et al., JINST 21 (2026) C01002, DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002)). This supports flexible forward modelling; it does not by itself make an arbitrary VDF uniquely identifiable from one measured projection.

References:
- PlasmaPy, “Thomson Scattering: Spectral Density,” https://docs.plasmapy.org/en/stable/notebooks/diagnostics/thomson.html
- Verdier, T. et al. (2025), DOI: 10.1063/5.0238948.
- Skolar, C. R., Longley, W. J. & Goodwin, L. V. (2025), Physics of Plasmas 32, 043907, DOI: 10.1063/5.0261966.
- Suazo Betancourt, J. L. et al. (2024), DOI: 10.1063/5.0158749.
- Eisenbach, S. et al. (2025), DOI: 10.1088/1361-6463/ad89d6.

## Legacy profile utilities

analyze_thomson_profile and reconstruct_thomson_profile remain downstream compatibility utilities for callers that already possess derived ne/Te profile points. The registered “Thomson spatial profile reconstruction” subanalysis is a distinct spectrum-to-profile path: it accepts measured, calibrated multi-channel spectra, runs the canonical physical inference, and then reconstructs every identifiable radial parameter profile using PCHIP and/or a generalized-cross-validation smoothing spline. It never accepts pre-derived ne/Te profile points. Full supplied cross-channel covariance is propagated through Monte Carlo where available; pointwise-only errors use an explicitly declared independent-channel approximation.

## Software testing

The new test module checks multi-channel data flow, arbitrary non-Gaussian response matrices, channel-specific wavelength axes, density identifiability, explicit ion-species requirements and uncertainty reporting. Synthetic tests are software regression tests, not independent experimental validation.

## Profile terminology and reconstruction

The profile summary reports radial-coordinate averages over the supplied interval. They are not line-of-sight averages or volume averages unless the corresponding geometric weighting is explicitly applied. The legacy attribute names containing "line_average" are compatibility aliases and emit DeprecationWarning when accessed.

The registered spectrum-to-profile subanalysis uses two numerical representations: shape-preserving PCHIP and a GCV smoothing spline. PCHIP is available with at least two distinct spatial channels; the smoothing spline requires at least five. The reconstruction is restricted to the measured radial interval and never extrapolates. Pointwise or full cross-channel covariance is propagated by Monte Carlo when supplied. These are spatial representations of parameters already inferred from measured spectra, not additional physical inference models.

Scale lengths are reported only where the gradient exceeds twice its propagated 1-sigma uncertainty when uncertainties are supplied; without uncertainty inputs, the numerical round-off threshold is used. The profile-summary function accepts separate full density and temperature covariance matrices and returns the corresponding full gradient covariance matrices. The downstream reconstruct_thomson_profile utility supports separate full density and temperature covariance matrices and a full joint density-temperature covariance matrix. The joint matrix is ordered [density_0..N-1, temperature_0..N-1] and is sampled jointly by Monte Carlo, preserving reconstructed ne–Te cross-covariance and returning reconstructed covariance and 95% bands. This legacy utility is downstream interpolation/smoothing of already-derived profile points. It is separate from the registered spectrum-to-profile workflow described above.


## Joint multi-channel inference and model-discrepancy uncertainty

The optional joint_spectral_covariance argument is a full positive-definite covariance matrix with dimension (n_channels * n_wavelengths, n_channels * n_wavelengths). Rows and columns use channel-major ordering: all wavelength samples for channel 0, then all samples for channel 1, and so on. It currently activates a joint generalized least-squares fit for:

- non_collective_maxwellian
- non_collective_kappa

The joint optimizer fits all channel-specific spectral parameters in one objective, whitens the complete residual vector with the Cholesky factor of the supplied covariance, and propagates the resulting parameter covariance to channel-resolved electron temperature and electron-flow projection. The output contains joint_inference.joint_parameter_covariance, physical_parameter_covariance, and total_physical_parameter_covariance. The final joint fit also refreshes each channel's integrated scattered power and, when absolute-density inputs are present, reruns the density inversion using that final area. Supplied laser/angle and LSF uncertainty contributions are retained when the joint covariance replaces the independent-channel fit covariance.

The collective model accepts optional 1-sigma arrays for ion_temperature_uncertainty_eV, ion_velocity_uncertainty_m_s, and electron_velocity_uncertainty_m_s, as well as the existing instrument_response_uncertainty and laser_wavelength_uncertainty_nm. When any are supplied, Monte Carlo refits perturb these external quantities and the response matrix, and return collective_external_parameter_covariance. This is not a full Bayesian marginalization: the result is conditional on the supplied uncertainty distributions, fixed ion identities/fractions and the selected Maxwellian collective model. Beam-direction uncertainty can be sampled component-wise; correlations among direction components and between distinct external sources are not represented.

Per-channel fits report a separate model_discrepancy_covariance when a reduced chi-square can be formed from supplied pointwise noise uncertainties and a valid fit covariance; otherwise they report residual/goodness-of-fit diagnostics only. Joint GLS reports its own excess-reduced-chi-square covariance component. This is a residual-based model-adequacy diagnostic; it does not prove that the physical forward model is complete or experimentally validated. Joint full-covariance GLS is not currently implemented for Collective, relativistic Maxwell-Juttner or custom models.

## Dashboard access

The Advanced Analysis menu exposes the registered Thomson spatial-profile subanalysis. It routes to the canonical calibrated-spectrum input workflow and runs the spectrum-to-profile reconstruction after physical inference; it does not create a second physical inference engine.
