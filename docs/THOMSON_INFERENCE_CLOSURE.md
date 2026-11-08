# Thomson inference closure: joint nuisance, profile and recoil contracts

This implementation adds lower-level, measured-spectrum inference primitives to
`tokamak_analysis.thomson_inference_closure` and public wrappers in
`tokamak_analysis.thomson_workflow`.

## Joint signal/background fit

`fit_thomson_spectrum_with_structured_background` fits the supplied physical
forward model and a detector-space background basis in one optimizer. The
returned covariance includes physical/background cross terms. Supply either
pointwise measurement uncertainty or a full positive-definite spectral
covariance. If neither is supplied, the fit is unweighted and any covariance is
residual-scaled; it is not measurement-noise uncertainty.

The routine does not decide which background basis is physically appropriate.
The caller supplies a measured/calibrated background design matrix. This avoids
silently treating a fitted continuum as plasma emission. The current
channel-by-channel Kappa and relativistic Maxwell-Juttner inference paths in
`thomson_inference.py` are not silently rewritten by this helper; callers must
pass the selected Kappa/MJ forward model explicitly. Full native Kappa/MJ
background integration across every UI/API path remains an integration task.

## Joint ne(r), Te(r), Ti(r) inversion

`reconstruct_joint_ne_te_ti_profiles` fits all supplied measured channel
spectra together and returns a joint profile covariance when the *data-only*
Jacobian has full column rank. It separately reports the rank of the
regularized Jacobian. A smooth solution whose data-only Jacobian is rank
deficient is labelled `non_identifiable_regularized_solution`; regularization
does not create measurement information.

The caller supplies the channel-resolved physical forward model. In
particular, Ti is not identifiable unless the supplied spectral model contains
an ion-sensitive feature. This primitive does not infer equilibrium mapping,
flux surfaces, PSF, channel cross-talk or ray geometry.

## Recoil and cross-section provenance

`thomson_compton_recoil_contract` records whether the relativistic model uses
elastic polarized Thomson scattering or single-scattering Compton kinematics
with polarized Klein-Nishina weighting. The recoil screen is an explicit
software threshold, not a universal plasma validity boundary. Multiple
scattering and unrestricted relativistic EVDF inversion are outside this
contract.

## Existing related implementations

- `cts_geometry_uncertainty.py` already performs full-forward Monte Carlo
  geometry refits. Each sampler draw must regenerate equilibrium, ray paths,
  beam overlap and scattering-volume quadrature; do not replace this with
  adding a final covariance matrix.
- `thomson_gap_closure.py` contains a relativistic 1-D projected EVDF inverse.
  It is not a general 3-D EVDF reconstruction.
- `thomson_population_mixture.py` contains a two-population MJ parametric
  approximation. It is not the generalized arbitrary-VDF algorithm of
  Skolar et al. (2025).
- The canonical Thomson UI and registry already expose the Compton-recoil
  selection for the relativistic MJ path. The recoil helper documents its
  cross-section semantics for programmatic callers.

## References

- Carole et al., *Journal of Instrumentation* **20** (2025) P07014,
  DOI: [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014).
- Skolar, Longley & Goodwin, *Physics of Plasmas* **32** (2025) 043907,
  DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966).
- Sakai et al., *Journal of Instrumentation* **21** (2026) C01002,
  DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002).


## Follow-on implementation: kinetic backgrounds, generalized projected VDF and workflow exposure

### Kappa and relativistic Maxwell-Juttner structured backgrounds

`tokamak_analysis.thomson_kinetic_background.fit_noncollective_kinetic_thomson_with_background`
provides a dedicated measured-spectrum API for the finite-variance isotropic
Kappa projection and relativistic Maxwell-Juttner forward model. The physical
parameters, spectral amplitude and detector-background coefficients are
optimized jointly. Its full local covariance includes the physical/background
cross block; it is conditional on the supplied measured response and noise
model. With no measurement uncertainty/covariance, the reported covariance is
residual-scaled and is not measurement-noise uncertainty. Absolute density is
not inferred. Kappa is screened by the declared non-relativistic limits
\(T_e/(m_ec^2)<0.01\) and \(|v_q|/c<0.1\).

This is a distinct lower-level workflow subanalysis and is now exposed in the
Advanced Analysis dashboard. It does not silently replace the existing
channel-wise physical-model selector; users must explicitly open the dedicated
Kappa/MJ structured-background subanalysis.

### Generalized tabulated electron VDF in electrostatic CTS

`tokamak_analysis.thomson_generalized_dispersion` adds a forward response for
a researcher-supplied tabulated 1-D electron VDF projected along the scattering
vector, with Maxwellian ion populations. The Landau integral primitive
analytically integrates the derivative of a piecewise-linear VDF for one
complex pole of any positive integer order per call, using a positive
imaginary pole displacement. The CTS susceptibility uses the first-order
pole. This is Skolar-inspired but does not implement multiple-pole summation
or the paper's complete kinetic closure. The forward model is
non-relativistic, unmagnetized and electrostatic; it does not infer an
unrestricted EVDF or its uncertainty. A measured-spectrum inverse must still
establish data-only identifiability before reporting distribution parameters.

### Full-forward ray geometry uncertainty

The CTS geometry Monte Carlo routine is now exposed through
`thomson_workflow.propagate_full_em_cts_ray_geometry_uncertainty` and included
in the canonical diagnostic registry. The sampler must jointly regenerate
equilibrium, probe/receiver rays, local beam covariance, scattering-volume
quadrature and overlap for each realization; every draw is then refitted through
the Full-EM spectral model. Geometry covariance is therefore not approximated
by a final-parameter covariance addition. Correlations with separately
propagated instrument-response uncertainty remain outside the joint sampler
unless the caller includes them in the same realization.

### Recoil contract and joint profile status

The Compton/recoil selector is exposed in the primary optical inference,
joint spatial-spectral path, relativistic multi-channel GLS and two-population
MJ dashboard and registry inputs. The output must distinguish polarized elastic Thomson
scattering from single-scattering Compton kinematics with polarized
Klein-Nishina weighting; neither branch claims multiple-scattering transport.

The joint \(n_e(r),T_e(r),T_i(r)\) closure now reports
`optimizer_failed` rather than a primary inference if the nonlinear
optimizer fails, even when a Jacobian rank calculation alone appears full
rank. Full cross-profile covariance remains conditional on convergence,
data-only full rank, the supplied forward model and the declared measurement
covariance. Regularization does not promote null-space components to primary
measurements.

## References for the added generalized response

- Skolar, C. R., Longley, W. J. & Goodwin, L. V. (2025), *Physics of Plasmas*
  **32**, 043907, DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966).
- Sakai, K. et al. (2026), *Journal of Instrumentation* **21**, C01002,
  DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002).


## 2026-11-03 numerical correction — projected EVDF velocity quadrature

The relativistic projected-1D optical Thomson inverse now includes explicit
trapezoidal integration weights over the supplied uniform velocity grid in its
forward operator. The modeled spectral density is therefore the discrete
approximation to S(lambda)=integral K(lambda,v) f_q(v) dv, rather than an
unweighted sum over velocity nodes. This removes the unphysical dependence of
the forward amplitude on the number of velocity-grid points. The endpoint
weights are half-interval weights, consistent with trapezoidal normalization
of the returned projected EVDF. The change does not relax the model's stated
1-D, symmetric-geometry, zero-recoil, fixed-background or fixed-response
assumptions. Synthetic regression coverage checks quadrature normalization
and uses the same integral convention when generating the test spectrum.


### Canonical channel-wise relativistic Maxwell–Jüttner background integration

The canonical `analyze_thomson_spectra` path now routes every relativistic
Maxwell–Jüttner channel through the joint physical/amplitude/background
variable-parameter fit after obtaining a deterministic seed from the established
relativistic fitter. It accepts the same explicit constant, linear, polynomial,
cubic-spline, and supplied-template detector-space backgrounds as the other
optical channel fitters. The reported covariance is the complete local
parameter covariance, including physical–amplitude, physical–background, and
amplitude–background cross terms. The integrated Thomson component and its
uncertainty are recomputed from the refitted forward model using the full joint
covariance; the detector background is excluded from the scattered-power
integral. The result remains conditional on the fixed measured response and
supplied measurement covariance, and does not infer absolute density without
the independent calibrated optical-power and collection inputs.


### Independent model-discrepancy covariance in the projected EVDF inverse

`reconstruct_relativistic_projected_thomson_evdf` now accepts an optional
wavelength-space `model_discrepancy_covariance`. The supplied matrix must be
finite, symmetric and positive semidefinite. It is added exactly once to the
independent signal and background measurement covariance components; the
effective covariance must be positive definite for GLS whitening. A
discrepancy-only weighting path is explicitly labelled as having no supplied
measurement-noise covariance. The result returns the measurement covariance,
discrepancy covariance and effective covariance separately, preserving their
provenance. The covariance remains conditional on the selected 1-D projected
EVDF, fixed response and fixed background; it is not a posterior uncertainty
or a correction to the physical scattering kernel.


## 2026-11-04 — joint nuisance covariance for structured-background Kappa/MJ fits

The dedicated Kappa/relativistic Maxwell–Jüttner measured-spectrum workflow now
accepts additive model_discrepancy_covariance and systematic_covariance
alongside either pointwise measurement uncertainty or a full measurement
covariance. Each additional covariance must be finite, symmetric and positive
semidefinite; the total covariance used for GLS whitening must be positive
definite. The inputs are added once, and the result records which components
were supplied. No missing covariance or cross-channel correlation is invented.

The fit now returns the local joint covariance ordered as
[Te, v_q, kappa (Kappa only), amplitude, background coefficients], including
the physical/background cross block and the background-coefficient covariance.
The profiled physical Jacobian and the complete physical-plus-nuisance Jacobian
are both checked for rank and conditioning. A deficient joint rank rejects the
fit rather than reporting a Kappa value that is degenerate with the selected
background basis. With no covariance inputs, the returned joint covariance is
residual-scaled and explicitly not measurement-noise uncertainty. The
covariance remains local and conditional on the fixed measured response and
selected background basis; absolute density is not inferred.

The canonical Dashboard subanalysis exposes uploads for model-discrepancy and
systematic covariance and forwards them to the same numerical entry point.
This closes the structured-background covariance gap for this dedicated
Kappa/MJ route only. It does not imply universal background parity in the
primary channel-wise Maxwellian/Kappa/MJ workflow or across CTS.


## 2026-11-05 — measured laser spectrum in joint spatial Compton inference

The wavelength-resolved joint spatial-profile inverse now supports a measured
incident-laser spectral profile when the relativistic Maxwell–Jüttner model
uses single-scattering Compton recoil. In this regime, incident wavelengths
are quadrature-integrated by evaluating the physical recoil-corrected,
polarized Klein–Nishina forward kernel at each measured source wavelength.
The resulting plasma spectrum is then passed through the measured detector
response. This deliberately replaces the elastic-Thomson wavelength-ratio
shift approximation, which is not valid for the nonlinear incident-photon
energy dependence of Compton recoil.

The optional two-column incident-laser profile upload is exposed in the
canonical Thomson Dashboard and declared in the spatial-profile diagnostic
contract. Without a measured laser profile, the source remains monochromatic
at the explicitly supplied laser wavelength. The source profile is normalized
for spectral-shape convolution; it does not provide absolute laser power or
electron density. This closes the measured-source-profile/Compton interaction
for the declared isotropic Maxwell–Jüttner single-scattering model only; it
does not extend the 1-D q-projection to a general 3-D EVDF or model multiple
scattering. The new regression checks that every source wavelength is evaluated
by the physical kernel and that the source quadrature is normalized.

## Multi-population electrostatic CTS input covariance

The electrostatic electrostatic_kappa fit accepts an optional
electron_population_parameter_covariance when electron_populations is
supplied. Its parameter order is population-major, with four parameters per
population:

1. density_fraction
2. temperature_ratio
3. kappa
4. flow_projection_m_s

The covariance must be finite, symmetric and positive semidefinite, with shape
(4 N_population, 4 N_population). Fraction covariance must preserve the
normalization constraint: every accepted realization has strictly positive
fractions summing to one. population_mc_samples (minimum 20) and
population_mc_seed control reproducible Monte Carlo sampling. Each accepted
draw is passed through a complete measured-spectrum CTS refit. Nonphysical
draws and failed fits are rejected; fractions are never renormalized and
physical parameters are never clipped. At least 70% of the requested refits,
and at least 20, must succeed or the uncertainty propagation fails explicitly.

fit_covariance_conditional_on_population_inputs preserves the original local
spectral-fit covariance conditional on fixed population parameters.
population_joint_fit_input_covariance reports the joint covariance of fitted
[n_e, T_e, kappa] and all population inputs, including their cross-covariance;
its fitted-parameter block adds the conditional fit covariance to the
Monte-Carlo population-sensitivity covariance. These are conditional
uncertainties under the supplied Gaussian input covariance and accepted-draw
policy, not a posterior or experimental validation. If no population
covariance is supplied, the prior fixed-input behavior is unchanged.


## Structured-background GLS fit acceptance and identifiability

The joint physical-spectrum/background GLS operator now distinguishes three
questions that must not be conflated:

- optimizer convergence;
- joint local identifiability of plasma and nuisance-background coefficients;
- physical-parameter identifiability after projecting the whitened physical
  Jacobian onto the orthogonal complement of the fitted background tangent
  space.

A covariance is returned only after optimizer convergence and full joint
Jacobian rank. An optimizer failure is explicitly marked
`not_reported_optimizer_failed`; a numerically full-rank Jacobian from a
non-converged iterate is not a valid uncertainty estimate. The result also
reports whitened residual chi-square, reduced chi-square, and a fit-acceptance
state that requires scientific residual review rather than automatically
declaring every converged fit acceptable. One-sided nuisance bounds receive a
feasible initial coefficient vector instead of an infeasible all-zero start.

## Finite velocity-support validity for projected EVDF inversion

The non-negative projected-EVDF inverse now reports an explicit finite-domain
tail diagnostic. For velocity nodes v_i, normalized nodal density f_i, and the
same nonuniform trapezoidal quadrature weights w_i used by the inverse, the
boundary probability is

    P_boundary = sum(i in {0, 1, N-2, N-1}) w_i f_i.

The default warning threshold is boundary_mass_fraction_limit=0.01. If
P_boundary exceeds the declared threshold, the result is marked
support_truncation_suspected; callers should expand the velocity domain and
repeat the inversion before interpreting tail-sensitive moments. This
diagnostic does not clip, extrapolate, or renormalize the reconstructed EVDF.

When the conditional normalized-EVDF covariance C_f is available, the
boundary-mass standard uncertainty is propagated as

    sigma(P_boundary) = sqrt(w_boundary.T @ C_f @ w_boundary).

The diagnostic remains separate from regularization-selection spread and
model-selection uncertainty. It is returned by the direct projected-EVDF
solver and by the Analysis-1 spectral-distribution path that invokes that
solver. The advanced Thomson physics Dashboard displays the boundary
probability and warns when the finite velocity support is suspect. This is a
numerical support-validity check, not a substitute for independent physical
validation of the inferred EVDF.
