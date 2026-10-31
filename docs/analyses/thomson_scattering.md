# Thomson Scattering — Analysis 1

## Primary contract

Analysis 1 starts from measured, wavelength-calibrated and intensity-calibrated Thomson spectra, not from electron-density or electron-temperature profiles. Its primary input is a complete spectrum for every spatial channel, the supplied spatial coordinate, and the measured instrument response. Detector calibration, raw-count correction and instrument calibration are outside this analysis.

The execution path is measured spectrum → selected physical forward model → measured-response convolution → physical parameter inference → channel-resolved diagnostics → optional spatial-profile reconstruction. No pre-derived \(n_e\) or \(T_e\) profile is accepted by the canonical analysis.

## Implemented model choices

The model must be selected explicitly in both the dashboard and the public Python API. The API rejects an omitted model; it never silently defaults to Maxwellian or substitutes one physical model for another.

### 1. Non-collective Maxwellian

A non-relativistic Maxwellian projected electron velocity distribution is fitted to the measured spectrum. The supplied instrument response is convolved with the calculated physical spectrum. The fitted centroid yields the electron-flow projection along the scattering vector.

### 2. Non-collective kappa

A finite-variance isotropic kappa distribution is represented by its one-dimensional Student-t projection. The analysis infers effective electron temperature and kappa index. This is a parameterized non-Maxwellian model, not an unrestricted multidimensional VDF reconstruction.

### 3. Non-collective relativistic Maxwell-Juttner

The relativistic single-particle model uses an isotropic Maxwell-Juttner electron distribution in its rest frame, Lorentz-boosted along the scattering vector. It evaluates the exact elastic Thomson photon-frequency ratio and Lorentz aberration, and transforms the explicitly supplied incident polarization into each electron rest frame before evaluating the polarized Thomson differential cross-section.

Required additional inputs are incident and scattered unit vectors consistent with the scattering angle, and an explicit incident polarization vector transverse to the incident beam. The inferred bulk-flow parameter is the projection along the scattering vector. Photon recoil/Compton corrections are neglected; this model is restricted to the Thomson limit. Arbitrary anisotropic relativistic optical VDF reconstruction is not claimed.

### 4. Collective Maxwellian

The collective Maxwellian model uses a kinetic dielectric/spectral-density calculation with explicit ion species, charge-density fractions, ion temperatures and ion/electron flow vectors. Collective density and temperature are inferred from the measured dielectric spectral shape. Ion temperatures, composition and flow vectors remain external inputs, not fitted parameters.

### 5. Collective kappa-electron dielectric model

The collective kappa model computes the unmagnetized longitudinal dielectric response from a one-dimensional projected isotropic kappa electron VDF and supplied Maxwellian ion species. Electron and ion susceptibilities are evaluated using the Landau-contour integral; the resulting dielectric function and dynamic structure factor are fitted to the measured collective spectrum. It infers \(n_e\), \(T_e\) and the kappa index, provided the three parameters are jointly identifiable.

This is a genuinely collective dielectric model, not a non-collective Student-t line-shape fit. It is limited to an unmagnetized longitudinal response with a one-dimensional projected isotropic kappa electron distribution and Maxwellian ions. Ion species, fractions, temperatures and flow vectors remain fixed external inputs. Magnetized or arbitrary 3-D non-Maxwellian collective optical inference is not claimed.

## Input contract

| Input | Unit / shape | Requirement |
|---|---|---|
| wavelength_nm | nm; wavelength or channel × wavelength | Required calibrated wavelength axis |
| spectra | calibrated intensity; channel × wavelength | Required measured spectra |
| radius_m | m; one per channel | Required supplied spatial coordinate |
| laser_wavelength_nm | nm | Required |
| scattering_angle_rad | rad | Required |
| instrument_response | measured kernel/matrix | Required measured response |
| signal_uncertainty | same unit/shape as spectra | Optional pointwise 1σ |
| joint_spectral_covariance | channel-major full covariance | Optional; joint GLS currently only for non-collective Maxwellian/Kappa |
| incident_direction / scattered_direction | dimensionless 3-vectors | Required for collective models and relativistic Maxwell-Juttner |
| incident_polarization | transverse dimensionless 3-vector | Required for relativistic Maxwell-Juttner; also used by absolute-density calculation |
| ion_species / ion_fractions | explicit metadata / charge-density fractions | Required for collective models |
| ion_temperature_eV | eV; one per species | Required for collective models; external input |
| ion_velocity_m_s / electron_velocity_m_s | m/s; explicit 3-vectors | Required for collective models; external inputs |
| collective_initial_density_m3 / collective_initial_temperature_eV | m⁻³ / eV | Required optimizer starting estimates for collective models |
| collective_density_bounds_m3 / collective_temperature_bounds_eV | m⁻³ / eV | Required explicit bounds for collective Maxwellian; optional for collective kappa |
| collective_kappa_initial | dimensionless; >1.5 | Required optimizer starting estimate for collective kappa |
| collective_kappa_bounds | dimensionless; lower bound >1.5 | Optional; if omitted, only the physical kappa >1.5 domain is enforced |
| absolute-density optical inputs | W, m, sr, efficiency and polarization geometry | Required for absolute non-collective density |

Input spectra are assumed to be prepared and calibrated. The analysis does not perform wavelength calibration, detector calibration, background correction of the measurement system, or instrument-response measurement. A local background nuisance parameter in the physical spectral fit is part of inference, not detector calibration.

## Instrument response

The measured response is applied in the forward model. Spectral-density samples are converted to bin-integrated power before applying the response and divided by measured-bin widths afterward, supporting nonuniform calibrated wavelength grids. A non-negative measured response matrix is mandatory; an identity matrix is accepted only when explicitly supplied and physically justified. No Gaussian line-spread function is imposed.

For non-collective Maxwellian/Kappa and relativistic Maxwell-Juttner fits, supplied response-matrix uncertainty is propagated by Monte Carlo perturbation and refitting. Both collective Maxwellian and collective kappa paths propagate supplied external-input and measured response-matrix uncertainties by Monte Carlo. The collective kappa path uses an unbounded positive log-parameterization for density and temperature unless the researcher supplies bounds; kappa is constrained only by its finite-variance physical domain unless explicit bounds are supplied. Local fit covariance is combined with the propagated external-input contribution. Correlations between different external inputs are not represented; no joint external-input covariance is assumed.

## Electron density and identifiability

A normalized spectral shape alone does not determine absolute electron density in the non-collective regime. Absolute density is calculated only when the spectral intensity is explicitly declared calibrated in W/nm and all required optical factors are supplied: laser power, interaction length, collection solid angle, optical efficiency, incident direction, incident polarization and scattered-light direction. If the contract is incomplete, density is reported unavailable rather than fabricated from fitted amplitude.

Collective Maxwellian and collective kappa models infer density from the dielectric spectral shape when the selected model is identifiable. A rank-deficient or ill-conditioned fit is rejected rather than returning an unsupported density or temperature.

## Outputs

- Channel-resolved electron temperature and uncertainty.
- Supported scattering-vector electron-flow projection and uncertainty.
- Electron density only where identifiable under the selected model and supplied absolute-power/collective conditions.
- Kappa index and uncertainty for non-collective and collective kappa models.
- Fitted spectra, residuals, local fit covariance, identifiability diagnostics and model provenance.
- Spectral-distribution diagnostics using the fitted background nuisance component.
- Spatial PCHIP/GCV reconstructions and covariance bands in the registered spectrum-to-profile subanalysis.

## Uncertainty and deliberate limitations

- Joint full-channel GLS with Cholesky whitening is currently implemented for non-collective Maxwellian and non-collective kappa models only.
- The relativistic Maxwell-Juttner model uses local fit covariance and response-matrix Monte Carlo where supplied. Laser-wavelength, scattering-angle and incident-polarization uncertainty propagation is not yet implemented for this model and is explicitly listed as missing.
- The collective kappa model uses local Jacobian covariance for \(n_e,T_e,\kappa\), with Monte Carlo propagation of supplied laser/angle, beam-direction, ion-temperature, ion-flow, electron-flow and response-matrix uncertainties. Cross-correlations among these external inputs are not represented; full channel-major GLS remains unavailable for this model.
- The collective Maxwellian model retains its existing supported external-parameter Monte Carlo. Its ion temperatures and flow vectors are still externally supplied, not fitted.
- The non-collective relativistic model assumes isotropic Maxwell-Juttner electrons and bulk flow parallel to the scattering vector. It does not infer an arbitrary anisotropic relativistic VDF.
- The collective kappa model is unmagnetized and longitudinal, with projected isotropic kappa electrons and Maxwellian ions. It does not implement magnetized or arbitrary 3-D non-Maxwellian collective optical scattering.
- The kappa models are parameterized distributions; they are not unrestricted VDF inversion.
- An integrated spectral integral is reported in watts only when the calibrated spectral-density unit is explicitly W/nm.
- No independent experimental validation is claimed. Automated tests are software/numerical consistency tests only.

## Spatial profile reconstruction

The registered “Thomson spatial profile reconstruction” subanalysis starts from measured calibrated multi-channel spectra, runs the canonical physical inference, and reconstructs every identifiable radial parameter profile using PCHIP and/or a generalized-cross-validation smoothing spline. It never accepts pre-derived \(n_e/T_e\) profile points and never extrapolates beyond the measured radial interval. Kappa profiles are included for both kappa model families.

For collective Maxwellian and collective-kappa profiles, the reconstruction covariance is the sum of (i) the cross-channel covariance from external-input Monte Carlo and (ii) the channel-local physical-fit covariance, including the separately reported residual-based discrepancy contribution when available. The diagonal therefore agrees with the per-channel total uncertainty used in the output; external-input covariance is not incorrectly treated as the complete uncertainty by itself. Electron-flow and ion-flow projection profiles retain only the covariance contributions actually estimated for those externally supplied flow vectors and geometry.

No public Analysis 1 or subanalysis entry point accepts already-derived electron-density or electron-temperature profile points. Spatial reconstruction always starts from measured calibrated multi-channel spectra and invokes the canonical physical forward-inference engine.

## Scientific basis and recent literature

The implementation uses a small set of physical model families rather than an unbounded collection of fitting options. Recent peer-reviewed references informing the current model scope include:

1. Carole et al., “A photonic model of the High-Resolution Thomson Scattering diagnostic of WEST tokamak,” *Journal of Instrumentation* **20** (2025), P07014, DOI: [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014).
2. Skolar, Longley and Goodwin, *Physics of Plasmas* **32** (2025), 043907, DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966), on non-Maxwellian electron-distribution effects in Thomson spectra.
3. Verdier et al., “Robust monitoring of thermal and fast ions using collective Thomson scattering,” *Review of Scientific Instruments* **96** (2025), 043513, DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948).
4. Sakai et al., *Journal of Instrumentation* **21** (2026), C01002, DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002), on Monte Carlo forward spectra for general electron distributions.
5. Foo et al., “Regularized inversion of optical Thomson scattering spectra for improved determination of plasma parameter evolution,” *Review of Scientific Instruments* **97** (2026), 103535, DOI: [10.1063/5.0350497](https://doi.org/10.1063/5.0350497).

These references motivate instrument-aware forward modelling, non-Maxwellian/relativistic spectral treatment and regularized inference. The TIS models are scoped implementations and are not claimed to reproduce every method in those papers.

## Dashboard access

The Spectral Analysis workspace exposes the canonical multi-channel Thomson inference. The Advanced Analysis workflow routes to the same canonical engine. The CTS microwave workflow separately offers electrostatic and magnetized full-EM models; it is not the optical Thomson Analysis 1 workflow.
