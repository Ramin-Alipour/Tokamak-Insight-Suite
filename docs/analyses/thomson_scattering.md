# Thomson Scattering — Analysis 1

## Primary contract

Analysis 1 starts from measured, wavelength-calibrated and intensity-calibrated optical Thomson spectra, not from electron-density or electron-temperature profiles. Its primary input is a complete spectrum for every spatial channel, the supplied spatial coordinate, and the measured instrument response. Detector calibration, raw-count correction and instrument calibration are outside this analysis.

The execution path is measured calibrated spectrum → explicitly selected physical forward model → measured-response convolution → physical parameter inference → channel-resolved diagnostics → optional spatial-profile reconstruction. No pre-derived \(n_e\) or \(T_e\) profile is accepted by the canonical analysis.

## Selected physical models

The model must be selected explicitly in both the dashboard and the public Python API. The current optical Analysis 1 shortlist contains exactly three models:

1. **Non-collective Maxwellian.** The standard incoherent single-particle Thomson spectral model for an isotropic non-relativistic Maxwellian electron distribution. The measured response is convolved with the calculated spectrum. The fitted centroid yields the electron-flow projection along the scattering vector. The classical model rejects inputs outside its declared non-relativistic temperature/flow regime.
2. **Non-collective Kappa.** A finite-variance isotropic Kappa electron distribution represented by its one-dimensional Student-t projection. The analysis infers effective electron temperature and the Kappa index. This is a parameterized non-Maxwellian model, not an unrestricted multidimensional velocity-distribution reconstruction.
3. **Non-collective relativistic Maxwell-Juttner.** An isotropic Maxwell-Juttner rest-frame electron distribution, Lorentz-boosted along the scattering vector. The forward model evaluates the exact elastic-Thomson photon-frequency ratio and Lorentz aberration, and transforms the supplied incident polarization into each electron rest frame before evaluating the polarized differential cross-section. It is the selected option for high-temperature regimes where the classical non-relativistic approximation is not appropriate. The code evaluates the maximum represented electron-rest-frame Compton correction fraction and rejects spectra exceeding the default 1% correction screen; the limit is configurable through the public spectrum model. This is an explicit numerical Thomson-limit screen, not a claim that recoil is modelled.

Optical collective dielectric models are not part of this three-model Analysis 1 shortlist. Microwave collective Thomson scattering (CTS) is another registered subanalysis within this same primary Analysis 1 family. It has its own explicitly selected electrostatic and magnetized full-electromagnetic forward models; these are CTS models, not additional optical models. No physical model is switched automatically.

### Recent peer-reviewed model and method basis

The three-model optical shortlist is deliberately limited to the common Maxwellian reference case, a finite-variance Kappa non-Maxwellian case, and a relativistic Maxwell-Juttner case. Recent literature supports these as useful, distinct modelling regimes without requiring a large catalogue of speculative models:

- **Maxwellian optical Thomson forward model:** Carole et al., “A photonic model of the High-Resolution Thomson Scattering diagnostic of WEST tokamak,” *Journal of Instrumentation* **20** (2025), P07014, DOI: https://doi.org/10.1088/1748-0221/20/07/P07014. This recent diagnostic paper describes an incoherent Thomson synthetic diagnostic and inference of electron temperature/density from measured polychromator signals.
- **Kappa / non-Maxwellian model family:** Skolar, Longley & Goodwin, “Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering,” *Physics of Plasmas* **32** (2025), 043907, DOI: https://doi.org/10.1063/5.0261966. It provides a numerical kinetic-integral method for Kappa and other non-Maxwellian distributions. The current TIS Kappa implementation is the explicitly documented analytic finite-variance Student-t projection; it must not be represented as implementing the Skolar generalized-dispersion algorithm.
- **Relativistic Maxwellian and Kappa spectral modelling:** Sakai et al., “Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions,” *Journal of Instrumentation* **21** (2026), C01002, DOI: https://doi.org/10.1088/1748-0221/21/01/C01002. The paper presents a Monte Carlo forward method and compares relativistic Maxwellian and Kappa spectra with analytical/numerical results. It supports the relevance of these regimes; it does not by itself validate the TIS implementation.

These papers inform model selection and implementation scope only. Their reported synthetic or experimental demonstrations are not validation of TIS. No additional model is selected solely because a paper proposes a new numerical optimizer, regularizer, or distribution parameterization.

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

## Incident-laser spectral profile

The canonical non-collective optical Thomson API optionally accepts
`laser_spectrum_wavelength_nm` and `laser_spectrum_density_per_nm`. These
arrays describe an already measured and calibrated incident-laser spectral
density; the software does not measure, calibrate, smooth, repair or infer the
source profile. The two arrays must be supplied together. The profile must be
non-negative, finite and strictly wavelength-ordered with positive integrated
area. Its shape is normalized to unit area because the source convolution
redistributes spectral shape, while absolute incident power remains a separate
input to the density inversion.

The forward ordering is

[
S_{\rm pred}(\lambda)=
R_{\rm inst}\left[
\int B_L(\lambda_L)
S_{\rm TS}(\lambda;\lambda_L)\,d\lambda_L
\right],
]

where (B_L) is the normalized measured laser spectrum,
(S_{\rm TS}) is the intrinsic Thomson spectrum evaluated at the incident
wavelength, and (R_{\rm inst}) is the supplied measured spectrometer response.
The implementation currently transfers the narrow-band incident wavelength
offset as an additive scattered-wavelength shift. This is the standard
narrow-band convolution approximation; it is not a full broadband
frequency-dependent scattering calculation. Finite-window source power that
shifts outside the measured wavelength interval is not folded back into the
spectrum. The output records source centroid, RMS linewidth, normalization
area, edge-column coverage and the approximation used.

If no measured laser profile is supplied, the model explicitly assumes a
monochromatic incident laser at `laser_wavelength_nm`. The source profile is
currently supported in the canonical non-collective Maxwellian, Kappa and
relativistic Maxwell-Jüttner fits, their joint spectral covariance/response
refits, the joint spatial-spectral profile inverse, and the projected
non-relativistic EVDF inverse. Collective dielectric optical inference is not
silently broadened by this approximation; it requires a source-frequency
resolved dielectric model and therefore rejects this optional input.

Recent X-ray Thomson-scattering literature also treats the source function and
instrument function as separate broadening operators and often combines them
into a source-and-instrument convolution only under a stated narrow-band
approximation. The present optical implementation follows that separation,
rather than adding source linewidth in quadrature to an inferred electron
temperature. Source-profile convolution is rejected when the optional Compton-recoil/Klein-Nishina mode is selected because a source-frequency-resolved recoil kernel is not yet implemented. See the 2026 XRTS source/instrument broadening review and the
2026 source-and-instrument function treatment (DOI:
10.1007/s41614-026-00227-9; 10.1063/5.0309991). These X-ray papers support the
general forward-model structure; they are not claimed as direct validation of
this optical wavelength-shift approximation.

## Instrument response and spectral-distribution diagnostics

The supplied measured response is applied in the physical forward model. Spectral-density samples are converted to bin-integrated power before applying the response and divided by measured-bin widths afterward, supporting nonuniform calibrated wavelength grids. No Gaussian line-spread function is imposed by the primary physical fit. A supplied 2-D or channel-specific 3-D measured response matrix must already have non-negative, unit-sum columns and is used unchanged. A supplied 1-D measured kernel must already have unit total area and is also used unchanged; when mapped onto the finite measured wavelength window, response tails outside that window are omitted, so edge columns can have sums below one. Those sub-unit edge sums represent spectral power redistributed outside the recorded wavelength interval, not a renormalization or correction of the calibrated kernel. The kernel path requires a uniform wavelength grid. Joint spectral GLS, optical-factor covariance and independent response-matrix Monte Carlo can be combined when the scattered-power component is excluded (zero first row/column) from the optical covariance to prevent double counting.

For the three optical models, supplied response-matrix uncertainty is propagated by Monte Carlo perturbation and refitting. A shared 2-D response matrix uses the same perturbation realization across all channels, and the resulting cross-channel covariance of temperature, flow projection and integrated power is returned; channel-specific 3-D response matrices retain channel-local uncertainty. The spectral-distribution subanalysis is separately callable (registry entry 81) but remains a child of primary Analysis 1; it requires the canonical physical fit to have completed for the same calibrated measured spectrum. It reports signed wavelength-space centroid, observed variance/width, skewness and excess kurtosis, together with a response-convolved Gaussian shape diagnostic. It does not claim to complete physical-parameter extraction when the canonical fit fails. It does not claim to reconstruct an arbitrary electron velocity distribution.

## Relativistic Compton option

The relativistic Maxwell-Juttner forward kernel exposes an explicit
`include_compton_recoil` switch. Its default is `false`: the existing
polarized elastic-Thomson model is retained and rejects a quadrature support
whose maximum fractional Compton energy correction exceeds
`maximum_recoil_correction_fraction`. When `true`, the kernel instead
uses the single-scattering Compton energy ratio
\[
r=\frac{\epsilon_s'}{\epsilon_i'}=
\frac{1}{1+(\epsilon_i'/m_ec^2)(1-\cos\Theta')}
\]
in the instantaneous electron rest frame. The scattered laboratory wavelength
includes the recoil factor (1/r), and the differential weight uses the
polarized Klein-Nishina factor, summed over final photon polarization states,
\[
\frac{1}{2}r^2\left(r+r^{-1}
-2|\mathbf e_i'\cdot\mathbf n_s'|^2\right).
\]
This expression reduces to the polarized Thomson factor
(1-|\mathbf e_i'\cdot\mathbf n_s'|^2) as (r\to1).
The option includes single-photon recoil and Klein-Nishina weighting; it does
not model multiple scattering, pair production, radiative corrections, or
collective photon effects. It is not enabled automatically. The fit and
multichannel APIs forward the same explicit switch. The reported effective
cross-section is correspondingly labelled as Thomson or polarized
Klein-Nishina according to the selected mode.

### Joint spatial-spectral Kappa inference

The joint spatial-spectral inverse accepts the finite-variance projected Kappa
kernel as a distinct physical model. By default, `kappa_index` is a fixed
external model parameter. Setting `fit_kappa_index=True` treats the supplied
`kappa_index` as a numerical starting value and estimates one shared Kappa
index jointly with the node-wise (T_e(r)), scattering-vector flow projection,
relative spectral amplitudes and detector-space background nuisance terms.
The transformed fit coordinate is (log(\kappa-1.5)), enforcing the
finite-variance condition (kappa>1.5). Its local covariance is transformed
back to physical Kappa units and returned as `kappa_index_uncertainty`.
The index is Primary only when the data-only Jacobian is full rank and the
selected model passes its validity screen. Otherwise it is not reported as a
Primary physical result. This remains a one-dimensional isotropic Kappa
projection with a single shared index; it is not the generalized plasma
dispersion algorithm of Skolar et al. and does not represent an arbitrary
anisotropic EVDF. Background basis coefficients remain nuisance parameters
and do not contribute artificial rank to the data-only identifiability test.

## Additive model-discrepancy covariance

The canonical channel-wise optical API
(`analyze_thomson_spectra`) accepts a researcher-supplied
`model_discrepancy_covariance` for each of the three selected non-collective
optical models: Maxwellian, isotropic Kappa and relativistic Maxwell-Juttner.
The accepted shapes are `(n_wavelength, n_wavelength)` (one common
within-channel covariance broadcast to every channel),
`(n_channels, n_wavelength, n_wavelength)` (channel-specific blocks), or
the full `(n_channels*n_wavelength, n_channels*n_wavelength)` channel-major
matrix. Values are in calibrated spectral-intensity units squared. Every
matrix must be finite, symmetric and positive semidefinite. The full
channel-major form preserves off-diagonal cross-channel model-discrepancy
blocks; it is added once to the supplied full measurement covariance. If
pointwise measurement uncertainties are supplied instead, their diagonal
variances are assembled in channel-major order before adding the full
discrepancy matrix. If no measurement uncertainty is supplied, the
discrepancy covariance itself must be positive definite for GLS weighting.
No silent diagonal jitter or eigenvalue clipping is applied.

The same additive rule applies to the full channel-major GLS paths for
Maxwellian, Kappa and relativistic Maxwell-Juttner models. The independent
block forms remain available for callers who only have within-channel
discrepancy estimates. No unspecified cross-channel correlations are inferred.

A separate joint structured-background Kappa/MJ workflow
(`fit_noncollective_kinetic_thomson_with_background`) accepts its own
wavelength-sized covariance for one spectrum and adds it to the supplied
measurement variance/covariance before GLS whitening. Its physical/background
nuisance treatment and covariance output are distinct from the canonical
channel-wise API. In every path, measurement-noise/model-discrepancy
cross-covariance is assumed zero, and resulting parameter uncertainties are
conditional on the supplied covariance model. Microwave CTS does not consume
this optical discrepancy argument; it requires a separately defined
CTS-specific discrepancy model.

## Electron density and identifiability

A normalized non-collective spectral shape alone does not determine absolute electron density. Absolute density is calculated only when the spectral intensity is explicitly declared calibrated in W/nm and all required optical factors are supplied: laser power at the scattering volume, interaction length, collection solid angle, optical efficiency, incident direction, incident polarization and scattered-light direction. The integrated scattered power is defined at the same calibrated measurement plane to which the end-to-end optical efficiency applies; efficiency is the fraction of power scattered into the collection solid angle that reaches that plane. Do not count the same transmission/throughput loss both in the measured power and in optical efficiency. The returned density is the interaction-length-averaged effective density; it equals a local density only for a uniform density/collection kernel along the scattering path. If the contract is incomplete, density is reported unavailable rather than fabricated from fitted amplitude.

For the relativistic Maxwell-Jüttner model, the density inversion does not substitute the stationary-electron classical differential cross-section. It uses the velocity-averaged effective polarized differential cross-section calculated by the same Lorentz-boosted Maxwell-Jüttner forward model, including the incident-flux and aberration factors. The fitted-parameter covariance and any supplied instrument-response, beam-geometry and polarization uncertainties are propagated through that effective cross-section before density uncertainty is reported. The fitted spectrum is normalized within the measured wavelength window; the inferred integrated power is corrected by the model's captured spectral probability so the density inversion uses total scattered power, not only the fraction inside the recorded window. Correlation between fitted power and effective cross-section is propagated in the density uncertainty.

The physical fit must pass its identifiability and fit-quality gates. Rank-deficient, ill-conditioned or unacceptable fits are rejected rather than returning unsupported physical parameters. No regime or model is silently inferred from missing metadata.

## Outputs

- Channel-resolved electron temperature and uncertainty.
- Electron-flow projection along the scattering vector and its uncertainty; no unsupported 3-D velocity reconstruction.
- Absolute electron density only where calibrated W/nm intensity and complete optical geometry/polarization make it identifiable.
- Kappa index and uncertainty for the non-collective Kappa model.
- Fitted spectra, residuals, local fit covariance, identifiability diagnostics and model provenance.
- Model-independent spectral-distribution diagnostics for each channel where the signed moments are identifiable.
- Spatial PCHIP and/or generalized-cross-validation spline reconstructions, with propagated uncertainty bands where the supplied covariance supports them.

## Recent optical Thomson developments

The 2025 WEST photonic-model paper describes a synthetic diagnostic spanning
the scattering spectrum, scattering volume, light collection, optical
components and polychromator response through to detector voltage. That is a
broader instrument-level forward chain than TIS's calibrated-spectrum
inference contract; TIS deliberately begins at measured calibrated spectra
and does not reproduce detector calibration or photon-transport calibration.
Tomita's 2025 perspective discusses practical laser Thomson scattering across
different plasma regimes and highlights the importance of optical design and
spectral analysis. Foo et al. (2026) present a regularized optical Thomson
inverse for improved temporal plasma-parameter reconstruction, reinforcing
the need to expose regularization, resolution and uncertainty assumptions
rather than report a unique distribution without qualification. These
publications motivate the source-profile option and the explicit limitations
of the projected-EVDF inverse; they do not constitute validation of TIS.

References:
- Carole et al. (2025), *JINST* 20, P07014,
  DOI: 10.1088/1748-0221/20/07/P07014.
- Tomita (2025), *Oyo Buturi* 94(9), 490–494,
  DOI: 10.11470/oubutsu.94.9_490.
- Foo et al. (2026), *Review of Scientific Instruments* 97, 103535,
  DOI: 10.1063/5.0350497.
- Sakai et al. (2026), *JINST* 21, C01002,
  DOI: 10.1088/1748-0221/21/01/C01002.

## Uncertainty and deliberate limitations

- Full channel-major generalized least squares with Cholesky whitening is implemented for non-collective Maxwellian, Kappa and relativistic Maxwell-Juttner fits. The joint fit uses the supplied complete covariance, including cross-channel measurement and (when supplied) model-discrepancy blocks, and retains channel-specific amplitudes and selected detector-space background nuisance coefficients. Independent channel fits are retained. Supplied laser-wavelength, scalar scattering-angle or beam-direction, incident-polarization and response-matrix uncertainties have separate Monte Carlo propagation routes.
- The relativistic Maxwell-Juttner external-parameter Monte Carlo samples the supplied 1σ component uncertainties and reprojects the polarization into the plane transverse to each sampled incident direction. Scalar angle uncertainty and beam-direction component uncertainties are alternative geometry descriptions and cannot be supplied together.
- Absolute-density uncertainty can combine the full channel-major spectral covariance, measured-response Monte Carlo and optical-factor covariance. When spectral/response uncertainty supplies the scattered-power term, the first row and column of the 6×6 (or 6N×6N) optical-input covariance must be zero; the measured-power contribution is then propagated separately without double counting. Scalar scattering-angle uncertainty is an additional common geometry contribution, including cross-channel covariance. Duplicate scalar optical-factor uncertainties remain prohibited.
- Kappa is a parameterized finite-variance distribution, not an unrestricted VDF inversion.
- The relativistic Maxwell-Juttner model assumes an isotropic rest-frame distribution and bulk flow parallel to the scattering vector. It does not include recoil; it screens the maximum represented rest-frame Compton correction and rejects cases above the configured 1% default limit.
- The non-relativistic Maxwellian and Kappa screens use Tₑ/(mₑc²) < 0.01 and |v_flow|/c < 0.1. These are conservative asymptotic error-budget screens (thermal relativistic corrections scale with θₑ; leading flow corrections scale with β²), not universal physical validity boundaries.
- Optical collective dielectric inference is outside the three-model optical shortlist. Microwave CTS is covered by its dedicated subanalysis within Analysis 1, with two separately selected CTS models.
- An integrated spectral integral is reported in watts only when the calibrated spectral-density unit is explicitly W/nm.
- No independent experimental validation is claimed. Automated tests are software/numerical consistency tests only.

### Input-data level and execution contract

The registry deliberately distinguishes `raw_data_to_physics=false` from `calibrated_measurements_to_physics=true`: the former excludes uncalibrated detector counts and raw acquisition signals; the latter describes inference from measured, calibrated spectra. These flags refer to different data levels and are not contradictory. The same rule applies to the microwave CTS subanalysis. The analysis does not calibrate the instrument or convert raw detector counts into calibrated spectra.

## Statistical weighting, background and spectral-distribution interpretation

Every fitted optical Thomson channel now reports `spectral_weighting_status` and `fit_covariance_interpretation`. A supplied pointwise uncertainty produces diagonal weighted least squares; a supplied channel-major covariance produces joint generalized least squares for Maxwellian/Kappa models. If neither is supplied, the fit is unweighted and its local covariance is residual-variance-scaled; it is **not** a measurement-noise-calibrated uncertainty. Do not interpret that covariance as a complete experimental uncertainty budget.

The channel-wise Maxwellian, Kappa and relativistic Maxwell-Juttner fits accept a researcher-selected detector-space nuisance basis: constant, linear, Legendre polynomial, cubic spline or supplied templates. In the joint channel-major GLS path each channel has its own coefficients and basis matrix, and the complete physical/background parameter covariance is retained. Basis adequacy and signal/background degeneracy remain researcher responsibilities; this is not an automatic physical decomposition of bremsstrahlung, ECE, line emission or other plasma backgrounds.

The spectral-distribution subanalysis is deliberately descriptive. It uses signed background-subtracted spectral measure for moments, retains negative residuals rather than clipping them, and reports the negative-area fraction. The moments require a positive signed integrated area and variance; strong cancellation or background error can make them unstable. Its Gaussian shape-fit covariance is conditional on the selected fixed background and fixed measured response. The result now explicitly reports whether the background was researcher-supplied or estimated from a selected edge fraction, and whether Gaussian fitting was weighted by measurement uncertainties. Edge-fraction estimation is a heuristic assumption, not an independently identified background. Moment uncertainty and Gaussian shape-fit uncertainty are separate from the physical parameter covariance of the parent Thomson fit.

## Spatial profile reconstruction

The registered “Thomson spatial profile reconstruction” subanalysis starts from measured calibrated multi-channel spectra, runs the canonical physical inference, and reconstructs every identifiable radial parameter profile using shape-preserving PCHIP and/or a generalized-cross-validation smoothing spline. It never accepts pre-derived \(n_e/T_e\) profile points and never extrapolates beyond the measured radial interval. The Kappa-index profile is included only for the non-collective Kappa model. The supplied radial coordinates must already be unique and strictly increasing; channel order is preserved exactly, and the workflow does not sort or reorder measured channels.

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

The Analysis 1 family contains four callable subanalyses: canonical multi-channel optical Thomson inference, microwave CTS, single-spectrum spectral-distribution diagnostics and spectrum-to-radial-profile reconstruction. The Advanced Analysis selector labels each child explicitly as an Analysis 1 subanalysis. CTS exposes its electrostatic and magnetized full-electromagnetic models without mixing them into the optical model shortlist.


## Single-spectrum spectral-distribution subanalysis (registry entry 81)

The callable `tokamak_analysis.thomson_distribution.analyze_thomson_spectral_distribution` is registered as a distinct entry point under primary Analysis 1. It calculates wavelength-space moments and a response-convolved Gaussian shape descriptor directly from one measured, calibrated optical Thomson spectrum. It does not accept pre-extracted physical parameters, independently refit the plasma model, or claim an arbitrary electron-velocity-distribution reconstruction.

The canonical multi-channel inference invokes this subanalysis for each channel using the same measured spectrum, the selected physical fit's background nuisance component, and the supplied measured instrument-response matrix. The physical electron temperature, identifiable electron density, and scattering-vector flow projection remain outputs of the canonical physical fit; the dashboard displays those values from the parent channel result alongside the spectral-distribution descriptors. A missing or rejected canonical physical fit prevents the dashboard subanalysis from being presented as a completed same-spectrum workflow. The parent API now marks both the spectral-distribution and projected-EVDF child results as `unavailable_primary_fit_rejected` when the selected physical fit fails its identifiability/fit-acceptance gate; it does not reuse nuisance-background coefficients from a rejected fit. Direct API calls to the descriptive distribution function remain independent measured-spectrum calculations and do not require a `physical_parameters` argument. Collective microwave CTS is handled by its dedicated subanalysis, not by this optical spectral-distribution routine.

The background for moments must be explicitly supplied as `background_signal`, `background_mask`, or a researcher-selected `background_fraction`; the code does not silently designate spectral edges as background. Measured response matrices are accepted only when their columns already satisfy the unit-sum bin-probability contract; non-normalized measured matrices are rejected rather than silently renormalized. The supplied calibrated response operator is otherwise used unchanged. Gaussian shape-fit parameter covariance is conditional on that fixed response matrix; response-matrix uncertainty is propagated for the canonical physical parameters by the primary fit, not re-estimated by this descriptive Gaussian fit. The legacy `thomson_spectrum.fit_thomson_spectrum` adapter also requires a measured response matrix and no longer synthesizes a Gaussian response from a scalar width.

The canonical channel-wise Kappa optimizer now uses a deterministic bounded multi-start least-squares search over six physically admissible combinations of initial Kappa index and spectral width (18 starts total). The selected solution minimizes the supplied-uncertainty-weighted residual sum of squares, or the unweighted residual sum of squares when no measurement uncertainty is supplied. The result records attempted/successful starts, selected objective value and optimizer termination metadata. These starts are numerical search coverage only; they are not Bayesian priors, physical bounds, or a guarantee that all local minima have been found.

The Kappa physical fit reports the Student-t probability captured by the measured wavelength window. It does not impose the Gaussian ±3σ coverage rule on a heavy-tailed distribution; if the fitted model has appreciable tails outside the measured window, the extrapolated full-model area is explicitly flagged. It also profiles the Kappa index while re-optimizing amplitude, centroid, width and linear-background nuisance parameters. The nominal one-parameter Δχ²=1 interval is conditional on the selected forward model and noise representation; grid-censored limits are not physical bounds. Without supplied measurement uncertainty, the profile uses an unweighted residual scale and is explicitly not measurement-noise calibrated.

## Literature traceability

The Kappa forward model implemented here is the analytic one-dimensional Student-t projection of an isotropic finite-variance Kappa velocity distribution. Skolar, Longley and Goodwin (2025, DOI 10.1063/5.0261966) develop a generalized plasma-dispersion-function approach for non-Maxwellian Thomson response; that is related literature but is **not** the same numerical algorithm as this Student-t fit. Sakai et al. (2026, DOI 10.1088/1748-0221/21/01/C01002) provide Monte Carlo reference spectra for arbitrary electron distributions, including relativistic Maxwellian and Kappa cases; TIS's deterministic quadrature is not claimed to reproduce their Monte Carlo method. These papers are references for comparison and equation traceability, not evidence of experimental validation of this repository.


## Recent literature and model provenance (2024–2026)

The selected optical forward-model family remains deliberately limited to three distributions: isotropic non-relativistic Maxwellian, finite-variance isotropic Kappa, and isotropic relativistic Maxwell–Jüttner. The 2026 incoherent-Thomson Monte Carlo study explicitly treats relativistic Maxwellian and Kappa distributions and compares the calculated spectra with analytical/numerical results; it is the recent literature anchor for retaining the Kappa and relativistic branches rather than adding a large catalogue of speculative distributions. The Maxwellian branch remains the standard reference model used in current fusion Thomson systems.

1. K. Sakai, K. Tomita, T. Hoshi and R. Yasuhara, “Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions,” *Journal of Instrumentation* **21** (2026), C01002. DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002). Supports incoherent spectral modelling for relativistic Maxwellian and Kappa electron distributions and discusses extension to arbitrary distributions; this software does not claim the unrestricted arbitrary-VDF inversion.
2. I. Yamada et al., “LHD Thomson Scattering Diagnostics,” *Journal of Fusion Energy* **44** (2025), 54. DOI: [10.1007/s10894-025-00520-4](https://doi.org/10.1007/s10894-025-00520-4). Recent fusion-device reference for electron-temperature/density profiles and studies of departures from a Maxwellian electron distribution.
3. M. Amarika et al., “ITER Core Plasma Thomson Scattering diagnostic design,” *Fusion Engineering and Design* **203** (2024), 114416. DOI: [10.1016/j.fusengdes.2024.114416](https://doi.org/10.1016/j.fusengdes.2024.114416). Recent ITER diagnostic reference for the primary electron-temperature and electron-density measurement objectives.
4. J. L. Flocken, S. B. Korsholm and J. Rasmussen, “Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas,” *Fusion Engineering and Design* (2026), 115845. DOI: [10.1016/j.fusengdes.2026.115845](https://doi.org/10.1016/j.fusengdes.2026.115845). Recent CTS reference for combining ray tracing, physical forward modelling and spectral inversion; it also identifies limits of cold-plasma ray tracing and the need to treat model/systematic uncertainties explicitly.
5. “The potential of collective Thomson scattering measurements in burning-plasma devices,” *Journal of Instrumentation* **21** (2026), C04045. DOI: [10.1088/1748-0221/21/04/C04045](https://doi.org/10.1088/1748-0221/21/04/C04045). Recent fusion CTS application reference covering thermal and energetic confined-ion measurements.

These recent papers are used as current application/model-scope references, not as a claim that every underlying classical Thomson or dielectric-response equation was first derived in 2024–2026. The implemented physical equations retain their established theoretical provenance; no new physical model is inferred from a recent application paper alone.


## One-dimensional electron velocity-distribution reconstruction (EVDF)

The spectral-distribution subanalysis now also exposes
`tokamak_analysis.thomson_evdf.reconstruct_thomson_evdf`. This is a
non-negative, second-difference Tikhonov inversion of the calibrated measured
spectrum. It maps a trial electron velocity component along the scattering
vector to its Doppler-shifted wavelength, applies the supplied measured
wavelength-response matrix as a forward operator, and fits the measured
spectrum after subtraction of the explicitly supplied background.

### Required inputs

- Measured, strictly increasing `wavelength_nm` and calibrated `signal`.
- `laser_wavelength_nm` and `scattering_angle_rad`.
- The measured `instrument_response_matrix`.
- An explicit `background_signal` on the same wavelength grid.
- Optional `regularization_strength`; when omitted, the solver selects the second-difference Tikhonov strength by generalized cross-validation (GCV) over its documented logarithmic candidate grid. A finite positive researcher-supplied value overrides GCV.
- Optionally, pointwise `signal_uncertainty` or a full within-spectrum
  `signal_covariance` (mutually exclusive).
- Optionally, pointwise `background_uncertainty` or a full
  `background_covariance` (mutually exclusive) for the explicitly supplied
  background.
- Optionally, `signal_background_cross_covariance` when both signal and
  background covariance models are supplied. The full joint block covariance
  must be positive definite. The residual covariance is evaluated as
  (C_{y-b}=C_y+C_b-C_{yb}-C_{yb}^{T}); no independence is assumed when a
  cross-covariance is supplied. If only background uncertainty is supplied,
  the result is conditional on the signal being treated as exact and is not a
  complete measurement-noise budget.
- Optionally, a velocity grid; otherwise the caller supplies its limit and
  point count.

### Identifiability and interpretation

For one optical scattering geometry, the primary reconstructed quantity is
the **normalized one-dimensional electron-velocity marginal along the
scattering vector**. The routine also reports its projected mean velocity,
variance, skewness, excess kurtosis and the directional equivalent temperature
\(m_e\,\mathrm{Var}(v_\parallel)/k_B\). This directional quantity is not
reported as scalar \(T_e\) unless isotropy is independently justified.

The inversion does **not** identify an unrestricted three-dimensional EVDF,
perpendicular velocity components, anisotropy, or absolute electron density
from one view. A multi-view reconstruction would require independent
scattering geometries and a separately validated tomographic forward model;
multiple views alone do not guarantee uniqueness. Absolute density requires
calibrated absolute scattered power and complete optical collection geometry,
which this normalized inversion does not calculate.

Alongside the non-parametric reconstruction, the routine fits and compares three parameterized projected-marginal candidates: Maxwellian, finite-variance Kappa/Student-t, and super-Gaussian. It reports weighted residual sum of squares, AIC and BIC for each successful fit. Candidate shape parameters are restricted only by mathematical admissibility: Student-t degrees of freedom \\(\\nu>2\\) for finite variance and super-Gaussian exponent \\(p>0\\) for normalizability. No finite upper cap is imposed without a researcher-supplied physical constraint. These candidate fits are distinct from the regularized basis reconstruction; no candidate is selected as physically true by the software.

When no regularization strength is supplied, the solver selects a value by generalized cross-validation (GCV) over a documented logarithmic candidate grid. The selected value, candidates and GCV scores are returned. An explicitly supplied positive value remains supported. A local active-set covariance is also estimated for the normalized EVDF and its pointwise standard uncertainty is returned. This is a conditional linearized covariance given the selected grid, response, background, regularization and active set; it is not a posterior distribution and does not include model-selection or unprovided systematic uncertainty. Sensitivity to the regularization and velocity grid must still be assessed before interpreting fine velocity-space structure. The solver now quantifies regularization sensitivity across every successful candidate in the same logarithmic GCV grid by reporting candidate-to-candidate changes in projected mean velocity, standard deviation, skewness, excess kurtosis and total-variation distance from the selected solution. If one positive regularization strength is explicitly supplied, the output identifies the single-candidate case and does not claim a sensitivity range. These are numerical robustness diagnostics, not measurement uncertainties or confidence intervals.

The primary Thomson spectrum-to-physics workflow automatically attempts this projected EVDF reconstruction only for the non-relativistic Maxwellian and Kappa optical models, using the same measured calibrated spectrum, fitted background nuisance component, measured response and supplied spectral uncertainty/covariance. The velocity interval is derived from the measured wavelength coverage and the supplied laser wavelength/scattering geometry; the grid size is configurable. A channel-level unavailable status and reason are returned if the spectrum does not support a stable inversion. When the relativistic Maxwell-Juttner model is selected, this non-relativistic inverse is explicitly withheld and the result is marked `unavailable_for_selected_physical_model`; a relativistically consistent projected-EVDF inverse is not implemented. This prevents a numerically converged but physically inconsistent non-relativistic inversion from being presented alongside relativistic primary results. It does not calibrate, correct, deconvolve, or otherwise prepare measured inputs.


### Fitted-background uncertainty in spectral moments

The descriptive spectral-distribution routine accepts either
`background_signal_uncertainty` (pointwise 1-sigma values) or
`background_signal_covariance` (a full positive-definite wavelength-space
covariance) when an explicit fitted `background_signal` is supplied. These
background nuisance perturbations are propagated jointly with any supplied
signal uncertainty through the moment Monte Carlo. The optional
`signal_background_cross_covariance` input represents
\(\operatorname{Cov}(s_i,b_j)\) for the measured signal and explicitly
supplied fitted background. It requires both marginal covariance models
(pointwise uncertainties are converted to diagonal covariance matrices) and
the complete \(2N\times2N\) block covariance must be positive definite.
Signal and background perturbations are then drawn jointly, preserving their
cross-correlation. If the cross-covariance is omitted, the two uncertainty
sources are treated as independent. The cross-covariance affects moment
uncertainty propagation only; the Gaussian shape fit remains weighted by the
signal covariance alone and the measured spectrum is not modified.
For mask- or edge-fitted backgrounds, the linear background is re-fitted in
each perturbed signal draw instead, so a separate background covariance or
signal/background cross-covariance input is rejected. Possible signal
contamination in a supplied background remains non-identifiable from this
subanalysis alone.


### Correlated-covariance spatial smoothing

When physical inference supplies a full positive-definite cross-channel parameter covariance, spatial reconstruction uses a natural-cubic smoothing-spline GLS objective,
\[
(\mathbf y-\mathbf f)^T\mathbf C^{-1}(\mathbf y-\mathbf f)
+\lambda\int [f''(r)]^2\,dr,
\]
with the smoothing parameter selected by generalized cross-validation using the effective degrees of freedom of the GLS hat matrix. The full covariance is used directly; correlations are never inferred from pointwise uncertainties. Results report the selected smoothing parameter, GCV score, effective degrees of freedom, generalized residual sum of squares and system condition number. Singular/non-positive-definite covariance is rejected rather than silently regularized. For positive profiles, the objective is applied to log-parameters with a first-order covariance transformation based on the measured channel values. Monte Carlo profile bands remain conditional on the supplied/inferred covariance and selected smoothing procedure.


### Kappa-shape identifiability

The non-collective Kappa fit evaluates the finite-difference Jacobian of the actual measured-response-convolved six-parameter spectrum. It reports numerical rank, normalized Jacobian conditioning, and the Kappa-shape sensitivity remaining after projection against amplitude, centroid, width and linear-background nuisance directions. The result includes an explicit `reportable_as_inferred_parameter` flag and a reason when the Kappa index is weakly/non-identifiable. In addition to the local Jacobian diagnostics, the fit performs a conditional profile-likelihood scan over the Kappa index, re-optimizing the five nuisance parameters at each fixed index. The reported interval is conditional on the selected physical model and supplied noise representation; a limit truncated by the finite scan grid is marked as censored rather than presented as a closed confidence interval. This is a model-conditional numerical identifiability diagnostic, not empirical validation.


## Joint wavelength-resolved spatial inference

The joint Maxwellian mode is exposed in the **Spectral Analysis → Thomson multi-channel spectral inference** dashboard. Select `non_collective_maxwellian`, enable the joint wavelength-resolved spatial option, upload the fixed detector-channel × radial-node spatial mixing matrix, and enter the numerical curvature-regularization strength explicitly. The matrix is a supplied forward operator; TIS does not estimate or calibrate it. The dashboard routes this selection through `reconstruct_thomson_profile_from_spectra` and exports the same profile-field names as the channel-wise workflow. Absolute density is returned as unavailable (`NaN`) in this shape-only joint inverse; it must not be inferred from relative spectral amplitude. Laser-wavelength, scattering-angle and spatial/spectral response uncertainty inputs are not propagated by this joint route; if supplied, the dashboard rejects the combination rather than silently dropping them. The result also exposes `parameter_identifiability`, and physical profile fields are withheld when the data-only Jacobian is rank deficient or the optimizer fails. A regularized optimizer candidate is retained separately for diagnostics and is never promoted to a measured result.

The optional `joint_spatial_spectral=True` mode fits the measured channel-by-wavelength spectra through a researcher-supplied fixed spatial mixing matrix and the supplied fixed spectral-response operator. It currently implements the isotropic, non-relativistic Maxwellian single-particle Thomson forward model; it does not estimate a point-spread function or calibrate either operator. The fit does not impose arbitrary physical bounds on electron temperature or projected flow: positive temperature and amplitude use only numerical logarithmic overflow guards, and fitted results are classified against the explicitly documented Maxwellian asymptotic-domain screen. The data-informed starting temperature is not clipped to a presumed scientific interval. The returned primary status is based on a finite-difference Jacobian of the whitened measured-data residual. Optional curvature-regularization rows are deliberately excluded from this rank test: regularization may select a smooth numerical solution, but it cannot create independent measurement information. Inspect `primary_result_status`, `primary_outputs_reportable_as_identified`, and `optimizer.data_only_jacobian_rank` before interpreting a reconstructed profile. If the data-only inverse is rank deficient or the optimizer does not converge, the physical profile and covariance fields are withheld (`null`); the optimizer's numerical candidate is retained separately under `optimizer_candidate` and explicitly marked as not a measured result. Perpendicular flow, null-space spatial structure, sub-resolution structure, and unrestricted anisotropic/non-Maxwellian distributions remain non-identifiable in this model.


## Path-integrated absolute-density inference

For a measured, absolutely calibrated scattered power and explicitly supplied
path-dependent optical factors, the API
`tokamak_analysis.thomson_scattering.infer_thomson_path_averaged_density`
evaluates

`P_s = integral n_e(s) P_i(s) [d sigma/d Omega](s) DeltaOmega(s) eta(s) ds`.

The output is the measurement-kernel-weighted path-average density, with
uncertainty from supplied scattered-power uncertainty. A single integrated
power measurement does **not** identify an arbitrary pointwise density
profile. The path-dependent inputs are treated as already characterized;
instrument calibration and experimental validation are outside this API.



The Streamlit Thomson workflow exposes this path-integrated capability as an
optional conditional calculation. The user supplies a path-profile CSV with
the named physical columns and an absolutely calibrated integrated scattered
power. No path-profile factor is calibrated or estimated by the software.


Instrument-response uncertainty may be supplied either as a non-negative 1-D
standard-uncertainty vector matching an explicit 1-D response kernel, a shared
2-D response-matrix uncertainty, or a channel-specific 3-D matrix uncertainty.
The 1-D kernel is perturbed, re-normalized through the same measured-response
forward operator, and the physical fit is repeated by Monte Carlo. An arbitrary
callable response has no supported uncertainty representation and is rejected
rather than assigned an invented covariance.


### Conditional uncertainty for projected-EVDF candidate models

Each successful Maxwellian, finite-variance Kappa/Student-t and super-Gaussian candidate fit now reports its local parameter covariance and 1-sigma parameter uncertainties when the fitted residual Jacobian has full column rank. If pointwise measurement uncertainty or a full spectral covariance is supplied, the covariance is conditional on that supplied noise model. Without measurement uncertainty, the covariance is residual-scaled and explicitly **not measurement-noise calibrated**. Parameters at an active physical/numerical bound have no symmetric interval and are marked boundary-limited; rank-deficient fits do not receive fabricated covariance values. These are local linearized uncertainties conditional on the selected candidate, velocity grid, measured response and fixed background, not model-selection uncertainty, posterior intervals, or experimental validation.


## Machine-readable Primary / Conditional / Non-identifiable result contract

The canonical optical Thomson API now includes a top-level `result_classification` object for every channel. It records the calibrated measured-spectrum input contract, explicitly states that no instrument calibration/correction or experimental validation was performed, and separates:

- **Primary:** electron temperature and scattering-vector velocity projection when returned by the selected physical fit; absolute density only when the calibrated absolute-power and optical-factor requirements are met.
- **Conditional:** absolute electron density when the required absolute spectral-power calibration, laser power, interaction length, collection solid angle, optical efficiency, beam geometry and polarization factors are not yet supplied; projected 1-D EVDF inference when the measured coverage/response/noise/model conditions do not permit it.
- **Non-identifiable:** unrestricted 3-D electron velocity distribution from a single scattering geometry. The measured spectrum constrains a projection along the scattering vector, not arbitrary perpendicular velocity structure.

Each classification entry includes the parameter name, dimensional unit where applicable, status/reason, and explicit auxiliary-input requirements. Numerical values remain in the existing channel/profile fields; the classification object does not duplicate or manufacture parameter values. This is a machine-readable scientific status contract, not a claim of experimental validation.


## Joint wavelength-resolved spatial inversion — response contract

The joint spatial-spectral Maxwellian inverse requires an explicit measured instrument-response operator. The API does not silently substitute an identity response when the argument is omitted. Supply the measured shared or channel-specific response matrix; an identity matrix is acceptable only when the researcher explicitly declares that the supplied spectra are already corrected for the instrument line-spread function. The operator is used unchanged and is never estimated or calibrated by TIS.


The Full-EM CTS inversion additionally supports conditional joint covariance propagation for explicitly supplied local magnetic-field, incident/scattered direction and complex polarization components. It uses Monte Carlo refits and is not available for frequency-resolved ray/equilibrium geometry; see [the CTS contract](collective_thomson_scattering.md) for the exact input names and limits.


## PSF-aware profile coordinate uncertainty

The standalone `reconstruct_profile_with_spatial_mixing` inverse accepts an
optional full covariance matrix `radius_covariance_m2` for the supplied spatial
node coordinates. It validates symmetry and positive semidefiniteness, samples
joint coordinate perturbations, rejects non-monotonic grids, and refits the
profile using the nominally selected regularization strength. When at least
80% of the requested draws (and at least 20) are valid, the coordinate-induced
profile covariance is added to the measurement-conditional profile covariance.
The result reports both covariance contributions, the number of successful
draws, and an explicit propagation status. If too few draws are valid, the
reported covariance remains conditional on the nominal coordinates and carries
a warning.

This extension propagates only the supplied coordinate covariance through the
fixed researcher-supplied spatial mixing operator and the selected curvature
regularization. It does not estimate the PSF, infer coordinate errors, propagate
uncertainty in the mixing operator, or close coordinate-uncertainty propagation
for the separate PCHIP/GCV profile workflow. Holding the regularization strength
fixed isolates coordinate uncertainty; variability from re-selecting GCV is
not included. This is numerical uncertainty propagation, not experimental
validation.


## Input-contract enforcement: calibrated spectra only

The Analysis 1 optical Thomson API accepts measured spectra that are already prepared and calibrated by the instrument/operator. Raw integer photon-count fitting is not part of this analysis contract. The former standalone `fit_optical_thomson_poisson_counts` entry point has been removed, together with its internal Poisson-likelihood branch and registry exposure. Use the canonical calibrated-intensity Gaussian/GLS inference path. This does not imply that calibrated spectral uncertainties must be Gaussian; the supplied uncertainty/covariance model is consumed according to the selected API. No count-to-power conversion, detector correction or calibration is performed by TIS.


## Spatial-coordinate uncertainty in profile reconstruction

The downstream profile-reconstruction API accepts the optional
`profile_coordinate_covariance_m2` argument: a finite, symmetric, positive
semidefinite covariance matrix with shape (N_channel, N_channel), in m² and
in the supplied channel order. It is propagated by Monte Carlo together with
the inferred profile-value covariance/uncertainty, when supplied. The
coordinate and profile-value perturbations are sampled independently because
the current input contract has no cross-covariance block between them; this
independence assumption is recorded in the result provenance. A draw is
discarded if perturbed coordinates are not strictly increasing or do not
cover the fixed reconstruction grid. The input coordinates are never sorted,
repaired or silently reordered. If the argument is omitted, coordinates are
treated as exact. This is uncertainty propagation for supplied spatial
coordinates, not an equilibrium or flux-surface mapping, and it does not
estimate coordinate uncertainty from the spectra.

## Calibrated forward-model comparison

The separate public helper
`tokamak_analysis.thomson_model_comparison.compare_thomson_forward_models`
compares at least two completed candidate spectra from the same calibrated
measurement. Supply the measured spectrum, a mapping of candidate model
predictions on exactly the same wavelength bins, and the number of fitted
parameters for each candidate. Supply either the full positive-definite
spectral covariance or pointwise 1-sigma uncertainty; if neither is available,
the helper uses a common unknown Gaussian variance estimated from residuals
and labels that assumption explicitly.

The result reports the weighted residual sum of squares, Gaussian
minus-two-log-likelihood, AIC, AICc, BIC and delta information criteria. The
unknown common variance is counted as an additional fitted parameter. AICc
is returned as unavailable when the sample size is insufficient for its
finite-sample correction.

This helper compares already-computed predictions: it does not perform the
Maxwellian/Kappa/Maxwell-Juttner nonlinear fits, estimate missing nuisance
parameters, select a model automatically, or establish physical truth.
Comparisons are meaningful only when candidates use the same measured bins,
likelihood, covariance convention and consistent treatment/counting of
nuisance parameters. Information criteria are model-comparison diagnostics,
not experimental validation.



## Relativistic multi-channel GLS and detector-background nuisance models

The channel-joint relativistic Maxwell-Juttner fit uses the complete supplied
positive-definite spectral covariance in channel-major ordering, including
non-zero cross-channel blocks. It shares electron temperature and the
scattering-vector flow projection across channels while retaining independent
channel amplitudes and detector-background coefficients.

The detector-space background basis is selected explicitly as
`constant`, `linear`, `polynomial`, `cubic_spline` or
researcher-supplied `templates`. The same basis is fitted simultaneously
with the relativistic plasma spectrum, after the supplied measured response
operator is applied to the plasma component. Shared templates or one template
matrix per channel are accepted. Background coefficients, their cross-terms
with the physical parameters, fitted spectra and the complete local
parameter covariance are returned. The basis is a nuisance model and is not
interpreted as plasma emission.

The common-Te/common-flow assumption, identical wavelength sampling and
scattering geometry, positive-definite covariance requirement, fixed measured
response, and single-scattering model assumptions remain in force. A
rank-deficient or ill-conditioned joint Jacobian is rejected. The reported
local covariance is conditional on the selected basis, supplied covariance,
response and physical model; it is not an experimental validation.


The same detector-background basis contract is used by the cross-channel
Maxwellian and finite-variance Kappa GLS refits. Their per-channel optimizer
seeds are converted back to the selected basis coefficients before the joint
fit; the joint parameter covariance retains physical/background cross-terms
and channel-to-channel correlations. This prevents the GLS refit from silently
replacing a selected polynomial, spline or template background with an affine
baseline.

### Explicit zero-background semantics for microwave CTS

The CTS physical-background radiative-transfer API now accepts exactly zero
electron and positive-ion density as an explicit vacuum/no-plasma state.
The vacuum consistency rule is enforced: a zero-electron-density
element must also have zero positive-ion density, and its supplied ECE
absorption coefficient must be exactly zero. For non-vacuum path elements,
quasineutrality is not silently imposed or repaired. The returned
`plasma_charge_state_moments_by_path` reports
`n_{e,quasi}=Σ_i Z_i n_i`, `Z_eff=Σ_i Z_i² n_i/n_e`,
`ρ_q=e(Σ_i Z_i n_i-n_e)` and the relative charge-balance residual.
`Z_eff` is unavailable when electron density is zero. The free-free
emissivity is exactly zero at zero electron density; no numerical floor is
converted into a spurious physical background.

The returned `background_status` distinguishes
`zero_background_exact_zero_radiance`,
`incident_radiance_only_no_plasma_background`, and
`plasma_emission_or_absorption_present`. The companion boolean
`zero_background` is true only when plasma free-free/ECE contributions and
incident radiance are all zero. A nonzero incident radiance in vacuum is
therefore preserved and is not mislabeled as a zero total signal. This
contract does not infer ECE opacity, plasma density, or a detector noise floor.


### Local cold-plasma magnetoionic modes and kinetic CTS boundaries

The advanced physics dispatcher exposes `cold_plasma_dielectric` and
`cold_plasma_magnetoionic_modes`. The local cold, collisionless,
multispecies dielectric uses the Stix functions
`R=1-Σ_s ω_{ps}²/[ω(ω+Ω_s)]`,
`L=1-Σ_s ω_{ps}²/[ω(ω-Ω_s)]`,
`S=(R+L)/2`, `D=(R-L)/2`, and
`P=1-Σ_s ω_{ps}²/ω²`, where
`Ω_s=q_s|B|/m_s` is signed. The magnetic-field vector sets both magnitude
and field-aligned basis. The zero-field limit is isotropic. For a supplied
wave-normal direction, the two local refractive-index branches are obtained
from
`det[ε_r-n²(I- k_hat k_hatᵀ)]=0`; complex roots and polarization eigenvectors
are returned. Cyclotron poles are rejected rather than assigned artificial
damping. This local cold-plasma model is not a warm kinetic dielectric,
Full-EM CTS three-wave vertex, ray tracer, or mode-conversion solver.

The electrostatic tabulated-fast-ion susceptibility interprets the finite
velocity grid as zero outside its support. Its Landau integral includes the
endpoint jump terms
`f(v_min)/(v_min-ζ)-f(v_max)/(v_max-ζ)`, as well as the interior principal
value and retarded Landau residue. The finite-support convention is therefore
explicit rather than dropping boundary contributions.

In Full-EM CTS, the measured real frequency shift is preserved in Maxwell's
wave tensor. The numerical resonance-broadening parameter enters kinetic
resonance denominators only; it does not floor or replace the real frequency.
An exactly zero frequency shift is currently unavailable because a separately
derived and converged static-Maxwell limit is not yet implemented.
