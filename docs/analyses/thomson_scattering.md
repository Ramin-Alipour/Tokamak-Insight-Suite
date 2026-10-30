# Thomson Scattering (Analysis 1 of 80)

## Primary contract

**Analysis 1 starts from measured, wavelength-calibrated and intensity-calibrated Thomson spectra, not from electron-density or electron-temperature profiles.** Its primary input is a complete spectrum for every spatial channel, with the channel's supplied spatial coordinate and the instrument response specified as a measured response matrix/function. Detector calibration and correction of raw detector counts are outside this analysis.

The primary execution path is:

1. Read the measured spectra as a channel-by-wavelength array (or long-form table in the dashboard).
2. Evaluate the selected physical Thomson forward model on the measured wavelength grid.
3. Convolve the physical spectrum with the supplied instrument line-spread response.
4. Fit each spatial channel and report channel-resolved physical parameters and diagnostics.
5. Construct the electron-temperature profile and, only when identifiable, the electron-density profile.
6. Keep the legacy calibrated-profile summary/reconstruction routines as downstream utilities; they are not the primary Analysis 1 input contract.

## Input contract

| Input | Unit / shape | Requirement |
|---|---|---|
| wavelength_nm | nm; (wavelength,) or (channel,wavelength) | Required calibrated wavelength axis |
| spectra | calibrated intensity; (channel,wavelength) | Required measured full Thomson spectra |
| radius_m | m; (channel,) | Required supplied coordinate for each channel |
| laser_wavelength_nm | nm | Required |
| scattering_angle_rad | rad | Required |
| instrument_response | callable, 1-D kernel, 2-D matrix or channel-specific 3-D matrix | Required measured response; use an explicit identity matrix only when a delta-like response is justified |
| signal_uncertainty | same unit/shape as spectra | Optional 1-sigma spectral uncertainty |
| instrument_response_uncertainty | same shape as response | Optional 1-sigma response-matrix uncertainty |
| ion_species | explicit species list | Required for collective Maxwellian mode |
| ion_fractions | dimensionless; one per species | Required for multiple ion species; fractions must sum to one |
| ion_temperature_eV | eV, one value per species | Required for the current collective backend; supply from an independent diagnostic when available |
| ion_velocity_m_s | m/s, one 3-vector per species | Required in collective mode; measured vectors or explicit zero vectors declaring stationary ions |
| electron_velocity_m_s | m/s, 3-vector | Required in collective mode; measured vector or explicit zero vector declaring stationary electrons |
| Absolute-density geometry | W, m, sr, dimensionless efficiency, incident direction, polarization and scattered direction vectors | Required with spectral intensity in W/nm to infer absolute density in the non-collective mode |
| Laser/angle uncertainty | nm / rad | Optional; missing uncertainty inputs are explicitly flagged |
| Optical-factor uncertainty | W, m, sr, dimensionless | Optional; only supplied uncertainty terms are propagated |

The measured spectra are assumed to be calibrated intensity data. No wavelength calibration, detector calibration, background correction of the measurement system, or instrument-response measurement is performed by this analysis. The physical fit may include a local background nuisance term, which is part of spectral inference rather than detector calibration.

## Implemented physical model families

### 1. Non-collective Maxwellian

The electron feature is represented by the non-relativistic Maxwellian Doppler spectrum. Its intrinsic width is fitted after applying the supplied instrument-response matrix in the forward model. The fitted spectral centroid yields the electron-flow projection along the scattering wavevector, not a full three-dimensional velocity vector.

Assumptions: non-collective single-particle scattering, isotropic Maxwellian electrons, non-relativistic velocities, and known scattering geometry.

### 2. Non-collective kappa (non-Maxwellian)

A finite-variance isotropic kappa electron velocity distribution is represented by its one-dimensional Student-t projection. The fitted second moment is mapped to an effective electron temperature; the kappa index is also returned. This is a parameterized non-Maxwellian option, not an arbitrary free-form velocity-distribution reconstruction.

Assumptions: non-collective scattering, isotropic kappa distribution, kappa index greater than 1.5, and non-relativistic velocities.

### 3. Relativistic Maxwell-Juttner (projected)

A relativistic option uses the one-dimensional marginal of an isotropic Maxwell-Juttner velocity distribution and a relativistic Doppler mapping along the scattering wavevector. This retains relativistic effects in the projected electron velocity distribution and the corresponding one-dimensional frequency shift. It is an explicitly reduced 1-D model: it does not integrate the full 3-D incident/scattered photon geometry and must not be interpreted as a complete relativistic Thomson-scattering solver.

### 3. Collective Maxwellian

The optional collective backend uses the PlasmaPy kinetic dielectric/spectral-density implementation. The model accepts explicit ion species, ion composition fractions, ion temperatures, ion flow vectors and an electron-fluid flow vector. Flow vectors must be independently measured or explicitly set to zero to declare a stationary-flow assumption. Missing ion temperatures are not silently replaced with electron temperature; multi-ion fractions are not silently assumed. If a required collective input is absent, the workflow stops with an explicit error. PlasmaPy and Astropy are optional dependencies installed with the plasma extra.

The collective model can cover both collective and non-collective scattering regimes through the dielectric spectral-density calculation. The scattering parameter and spectral shape depend on the supplied plasma parameters and geometry.

## Instrument response

The instrument response is represented by a non-negative matrix whose columns are normalized probability distributions for detected wavelength bins conditional on each true wavelength bin. The measured response is mandatory and may be supplied as a callable evaluated on the wavelength grid, a 1-D odd-length kernel, a 2-D wavelength-to-wavelength matrix, or a channel-specific 3-D matrix. The same response can be used for all channels, or a separate matrix can be supplied per channel. An identity matrix is accepted only when explicitly supplied and physically justified. This supports asymmetric and wavelength-dependent line-spread functions; a Gaussian response is not imposed.

For the non-collective Maxwellian, kappa and projected relativistic fits, supplied response-matrix uncertainty is propagated by Monte Carlo perturbation of the response matrix, followed by refitting. The reported temperature and velocity uncertainties combine fit covariance and the Monte Carlo response contribution when enough valid perturbation fits are obtained. The collective backend currently does not marginalize response uncertainty into its parameter covariance.

## Electron density and identifiability

A spectral shape alone does not determine absolute electron density in the non-collective regime. Absolute density is calculated only when the spectral intensity is explicitly declared calibrated in W/nm and all required optical factors are supplied:

- incident laser power at the scattering volume;
- interaction length;
- collection solid angle;
- optical efficiency;
- incident laser propagation direction;
- incident polarization vector (perpendicular to the incident beam);
- scattered-light direction (consistent with the specified scattering angle).

The differential Thomson cross-section uses the incident polarization dependence for an unanalysed outgoing polarization measurement. The incident polarization must be perpendicular to the incident beam, and the supplied incident/scattered direction vectors must agree with the specified scattering angle. If the absolute-power contract is incomplete, the density profile is reported as unavailable (NaN) with an explicit status; it is not fabricated from the fitted Gaussian amplitude. First-order density uncertainty propagation includes only uncertainty terms supplied by the user (spectral integrated-power fit uncertainty and the listed optical-factor uncertainties). Each channel lists unavailable uncertainty inputs and marks the uncertainty completeness as partial; missing uncertainties are never silently treated as zero.

## Outputs

- Per-channel fitted electron temperature and its uncertainty.
- Per-channel electron-flow projection along the scattering wavevector and its uncertainty, where supported.
- Per-channel absolute electron density only when identifiable from the selected model and supplied optical inputs.
- Multi-channel radial profiles of the inferred quantities.
- Per-channel fitted spectra, residuals, fit covariance and model assumptions.
- Explicit model, instrument-response usage, input provenance and limitations.

## Deliberate limitations

- The relativistic Maxwell-Juttner option is a 1-D projected-velocity approximation. Full 3-D relativistic photon-geometry integration is not implemented.
- Collective non-Maxwellian kinetic distributions are not implemented; the collective backend is Maxwellian. The non-collective kappa option is parameterized and does not reconstruct an arbitrary VDF.
- The kappa model is a parameterized isotropic non-Maxwellian approximation, not an arbitrary electron VDF inversion.
- Absolute density in non-collective mode requires absolute W/nm intensity and complete optical geometry; spectral shape alone is insufficient.
- The current collective fit reports a local least-squares covariance for fitted density and temperature, but does not marginalize uncertainty in ion temperatures, ion flows, electron flows or the instrument response.
- Correlated uncertainty between wavelength bins, calibration factors and channels is not modeled.
- No independent experimental validation is claimed. Automated tests verify software behavior and mathematical consistency only.

## Scientific basis and model selection

The implementation uses a small set of model families rather than an unbounded list of fitting options:

1. The Maxwellian collective/non-collective spectral-density formulation follows the kinetic dielectric treatment documented by PlasmaPy. Its spectral-density function explicitly supports one or more ion species, ion temperatures and drift velocities.
2. Recent experimental CTS work demonstrates the importance of robust spectral/background treatment for thermal- and fast-ion monitoring (Verdier et al., Review of Scientific Instruments 96, 043513, 2025, DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948)).
3. Recent non-Maxwellian Thomson forward-model work motivates retaining a non-Maxwellian option rather than treating a Gaussian as a universal spectrum (Skolar, Longley & Goodwin, Physics of Plasmas 32, 043907 (2025), DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966)). The kappa implementation here is a bounded parameterized approximation and is not claimed to reproduce that paper's general arbitrary-distribution solver.
4. Recent multi-regime analysis of Thomson spectra reports inference in both collective and non-collective regimes using physical spectral models (Eisenbach et al., Journal of Physics D: Applied Physics 58, 035202, 2025, DOI: [10.1088/1361-6463/ad89d6](https://doi.org/10.1088/1361-6463/ad89d6)).

References:
- PlasmaPy, “Thomson Scattering: Spectral Density,” https://docs.plasmapy.org/en/stable/notebooks/diagnostics/thomson.html
- Verdier, T. et al. (2025), DOI: 10.1063/5.0238948.
- Skolar, C. R., Longley, W. J. & Goodwin, L. V. (2025), Physics of Plasmas 32, 043907, DOI: 10.1063/5.0261966.
- Suazo Betancourt, J. L. et al. (2024), DOI: 10.1063/5.0158749.
- Eisenbach, S. et al. (2025), DOI: 10.1088/1361-6463/ad89d6.

## Legacy profile utilities

analyze_thomson_profile and reconstruct_thomson_profile remain available for users who already have independently obtained calibrated density/temperature profiles. They are downstream profile-summary utilities only and are no longer the primary Analysis 1 contract or registered entry point.

## Software testing

The new test module checks multi-channel data flow, arbitrary non-Gaussian response matrices, channel-specific wavelength axes, density identifiability, explicit ion-species requirements and uncertainty reporting. Synthetic tests are software regression tests, not independent experimental validation.
