# Advanced CTS and optical Thomson operators

This document is the authoritative scope note for advanced Analysis 1 operators.
All physical inference starts from measured, prepared and calibrated diagnostic
data. TIS does not calibrate instruments, repair detector data, estimate a
missing instrument response, or perform experimental validation.

## Calibrated Optical Thomson inference

The canonical optical path accepts calibrated measured spectra, a wavelength
axis, the selected scattering geometry, and the measured instrument-response
operator. The researcher explicitly selects one of three implemented
non-collective forward models:

1. Isotropic non-relativistic Maxwellian.
2. Isotropic finite-variance Kappa, represented by its analytic one-dimensional
   Student-t projection.
3. Isotropic relativistic Maxwell-Juttner, with the declared elastic Thomson
   or single-scattering recoil/Klein-Nishina option.

The canonical optical inference uses Gaussian/GLS residuals. There is no public
raw integer photon-count/Poisson-likelihood input path in Analysis 1. Do not
pass raw counts to the calibrated-spectrum APIs and do not describe the
retired count-space route as a supported capability.

### Model-discrepancy covariance

The canonical single-channel Maxwellian, Kappa and relativistic
Maxwell-Juttner optical fits accept a researcher-supplied wavelength-space
`model_discrepancy_covariance`. It is an independent additive covariance
component and is combined with the supplied measurement covariance before
GLS whitening. It is not estimated from reduced chi-square.

For multi-channel inference, the supported discrepancy representation is
shared wavelength covariance or one within-channel covariance block per
channel. With a full channel-major measurement covariance, cross-channel
measurement covariance is retained, while the supplied discrepancy blocks
are added to the corresponding diagonal channel blocks. Arbitrary
cross-channel discrepancy covariance and measurement-noise/discrepancy
cross-covariance are not inferred.

The separate structured-background kinetic workflow for Kappa and
Maxwell-Juttner also accepts its own single-spectrum discrepancy covariance.
This is not a universal covariance API for all CTS branches.

### Model comparison

`tokamak_analysis.thomson_model_comparison.compare_thomson_forward_models`
compares at least two already-computed candidate forward spectra on identical
calibrated wavelength bins and under one common Gaussian likelihood. It reports
the weighted residual objective, AIC, AICc and BIC, with the appropriate
common unknown-variance parameter counted when no measurement covariance is
provided.

This helper does not refit model parameters, select a physical model
automatically, establish physical truth, or validate a model. Candidate
predictions and their fitted-parameter counts must come from the researcher's
completed fits. Information criteria are only comparable when the data bins,
likelihood, covariance convention and treatment of nuisance parameters are
consistent.

## Spectral-distribution subanalysis

`analyze_thomson_spectral_distribution` reports descriptive wavelength-space
moments and response-convolved Gaussian shape-fit diagnostics. It is not a
substitute for the physical Maxwellian/Kappa/Maxwell-Juttner inference and does
not identify an electron velocity distribution from moments alone.

Exactly one background representation is explicit: a supplied fitted
background spectrum, a researcher-supplied mask, or an explicitly selected
spectral-edge fraction. No spectral-edge region is silently assumed. Signed
background-subtracted samples are retained for moment calculations; negative
samples are not silently clipped. Moment outputs describe the measured
wavelength-space signal and are not intrinsic thermal widths or electron
temperatures.

## Spatial profile reconstruction

The channel-wise workflow first infers identifiable physical parameters from
each calibrated spectrum, then reconstructs their radial profiles using
PCHIP and/or generalized-cross-validation smoothing splines with the
available parameter covariance.

The joint spatial-spectral inverse is a distinct route: it fits all measured
channel-by-wavelength spectra through a researcher-supplied fixed spatial
mixing operator and measured spectral-response operator. It supports the
declared Maxwellian, finite-variance Kappa and relativistic Maxwell-Juttner
kernels. It does not estimate or calibrate the mixing operator/PSF.

The PSF-aware profile inverse operates on already inferred channel values and
a fixed researcher-supplied mixing matrix. It is not a joint fit to the raw
spectra. It reports data-only rank and suppresses a unique primary profile
when the latent spatial nodes are not identifiable. Its conditional
regularized covariance excludes uncertainty in the supplied mixing operator.

Coordinate covariance is supported in the downstream profile reconstruction
path where declared. Coordinate/value cross-covariance and automatic
equilibrium or flux-coordinate mapping are not inferred. Interpolated or
regularized nodes are not independent measurements.

## Electrostatic CTS

Electrostatic CTS is a microwave collective-scattering model, not an optical
Thomson model. It uses the longitudinal dielectric response and requires
explicit ion species, charge states/fractions, geometry and calibrated
frequency-shifted spectral power. It supports Maxwellian thermal species and
the documented projected fast-ion distribution options.

The optional reference-spectrum PCA background requires at least two supplied
calibrated background-only spectra on the same frequency grid and detector
intensity scale. The retained component count is selected explicitly by the
researcher. The finite-reference bootstrap can quantify reference-basis
uncertainty and propagate it through repeated plasma-parameter refits. This is
not claimed to reproduce the modulation-aware physics-informed PCA algorithm
of Verdier et al. (2025).

A correlated additive `model_discrepancy_covariance` is supported in the
electrostatic CTS likelihood. It is not silently passed into the Full-EM
solver. The discrepancy covariance is independent of measurement covariance;
cross-covariance between those sources is not inferred.

## Full-electromagnetic CTS

The Full-EM branch uses the magnetized kinetic electromagnetic response,
species-resolved thermal-ion response, fluctuation covariance, and the
documented nonlinear scattering closure. Explicit relativistic gyrotropic
electron/fast-ion distributions are supported by the forward spectral-density
model and by the separate conditional population-shape inversion.

The main measured-spectrum inversion has its own input contract. It does not
silently infer an arbitrary drifted relativistic electron VDF. The separate
shape inversion is conditional on fixed bulk plasma parameters, geometry,
polarization, response and population density; a rank-deficient result is a
regularized candidate, not a uniquely identified distribution.

The Full-EM measured-spectrum inversion fits scalar flow projections along
the scattering wavevector, not unrestricted three-dimensional flow vectors.
It reports local fit covariance, identifiability diagnostics and numerical
harmonic/quadrature convergence. Numerical convergence is a discretization
check, not physical-model uncertainty or experimental validation.

Cold-plasma ray tracing and paraxial beam covariance are conditional
geometrical-optics tools. Branch tracking and the optional lossless
coupled-mode propagation utility do not constitute a complete dissipative
full-wave mode-conversion solver. Non-paraxial diffraction, general
non-Gaussian beam evolution, absorption/collisional mode coupling and
arbitrary inhomogeneous wave propagation are not implied.

The Full-EM measured-spectrum inversion accepts an optional independent
additive `model_discrepancy_covariance` on the measured frequency-shift grid.
It combines this with either the supplied full measurement covariance or the
diagonal covariance formed from pointwise measurement uncertainty. A
discrepancy-only covariance is permitted and is labelled as such; without
either source the fit remains unweighted. The discrepancy covariance must be
positive semidefinite and the total covariance must be positive definite for
Cholesky GLS. No jitter, eigenvalue clipping or residual-derived discrepancy
is introduced. Measurement/discrepancy cross-covariance and
discrepancy-hyperparameter inference are not included.

The Full-EM measured-spectrum inversion currently fits spatially uniform
plasma parameters over the sampled scattering volume. Spatially varying
magnetic field and beam geometry do not imply that arbitrary node-wise
`n_e`, `T_e` and species-resolved `T_i` profiles are being fitted. A
profile-varying kinetic inversion requires evaluating the local kinetic
forward response at each spatial quadrature point and jointly identifying
the profile degrees of freedom.

Thermal free-free background and LTE ECE transfer are available as separate
forward radiative-transfer operators. ECE requires an externally supplied
mode-resolved absorption coefficient from a suitable magnetized dielectric
calculation; it is not fabricated from `B`, `n_e` and `T_e` alone.
Radiance-to-receiver-power conversion requires supplied receiver etendue and
end-to-end efficiency. The resulting physical background is an explicit
receiver-plane input to the Full-EM spectral inversion.

## Deliberate scientific boundaries

The following are not claimed as completed universal capabilities:

- Arbitrary non-Maxwellian optical Thomson inversion using a general
  generalized-Landau kernel for every supplied distribution.
- Unrestricted drifted relativistic electron-VDF inference in the main
  Full-EM measured-spectrum optimizer.
- Automatic inference of ion composition, fast-ion populations, PSF,
  equilibrium, ray geometry, polarization or missing radiative-transfer data.
- Finite-orbit-width corrections or a general collisional kinetic response
  across all CTS regimes.
- Full-vector, non-paraxial, dissipative quasi-optical propagation.
- Full covariance propagation for every geometry, profile, model-discrepancy
  and nuisance-input source, including all cross-source correlations.

These boundaries must remain explicit until a physically derived model,
well-defined measured-input contract, numerical implementation and regression
tests are added. No synthetic software test is experimental validation.

## Recent peer-reviewed method context

- Carole et al. (2025), *Journal of Instrumentation* 20, P07014,
  DOI: 10.1088/1748-0221/20/07/P07014 — high-resolution optical Thomson
  spectral modelling.
- Skolar, Longley & Goodwin (2025), *Physics of Plasmas* 32, 043907,
  DOI: 10.1063/5.0261966 — generalized plasma-dispersion treatment for
  non-Maxwellian plasmas and Thomson scattering.
- Sakai et al. (2026), *Journal of Instrumentation* 21, C01002,
  DOI: 10.1088/1748-0221/21/01/C01002 — Monte Carlo Thomson spectra from
  arbitrary electron distributions.
- Verdier et al. (2025), *Review of Scientific Instruments* 96, 043513,
  DOI: 10.1063/5.0238948 — CTS thermal/fast-ion background estimation.
- Flocken, Korsholm & Rasmussen (2026), *Fusion Engineering and Design*,
  DOI: 10.1016/j.fusengdes.2026.115845 — accuracy-driven CTS modelling,
  ray tracing and Bayesian inversion.

The recent papers guide selected extensions; they do not replace the
underlying Thomson cross-section, kinetic dielectric response or
three-wave-coupling equations. No experimental validation is performed or
claimed.


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


## Unified public Analysis-1 solver entry point

The four advanced solver paths are available through
tokamak_analysis.cts_full_physics.run_advanced_thomson_physics(model, **kwargs).
The explicit model keys are:

- landau_fdt_cts
- polarized_kinetic_ece
- stratified_nonparaxial_vector
- stratified_mode_conversion

The dispatcher forwards the caller's inputs to exactly one solver, preserves
the solver's limitations and convergence metadata, and never substitutes a
simpler model. This is an API/integration improvement; it does not expand the
physical validity domains documented above. The Landau-FDT branch remains a
finite-box, non-relativistic, isotropic Maxwellian electrostatic CTS reference
solver; kinetic ECE remains non-relativistic Maxwellian; vector propagation
and mode conversion remain restricted to the documented stratified geometry.
