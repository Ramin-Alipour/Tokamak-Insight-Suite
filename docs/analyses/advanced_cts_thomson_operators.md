# Advanced CTS and optical Thomson inverse operators

This page records the scientific scope of the advanced operators added to
Analysis 1. All measured inputs remain researcher-prepared and calibrated.
No detector calibration, data repair, empirical validation or synthetic-to-
experimental equivalence is claimed.

## Optical Thomson: independent Poisson likelihood

The separate public API
`fit_optical_thomson_poisson_counts` uses signed square-root Poisson-deviance
residuals for raw integer photon-count bins. It requires a uniform wavelength
grid and a measured instrument-response matrix. It returns count-space
observables, including captured fitted signal counts, and does not report
watts or infer absolute electron density.

This API is currently restricted to the non-collective Maxwellian model.
Kappa, relativistic Maxwell-Jüttner and collective fits retain their existing
Gaussian/GLS likelihood paths. Do not pass calibrated power-density data to
the Poisson API.

## Independent model discrepancy

`gaussian_process_discrepancy_covariance` constructs a correlated additive
covariance using squared-exponential, Matérn-3/2 or Matérn-5/2 kernels.
Amplitude, length scale and nugget are explicit inputs; no values are inferred
from reduced chi-square. Electrostatic CTS accepts this covariance through
`model_discrepancy_covariance` and adds it to the measurement covariance
before GLS whitening. When supplied, the legacy reduced-chi-square
overdispersion inflation is not added on top.

The GP helper does not estimate hyperparameters. They must be selected using a
declared likelihood/model-comparison procedure. The same covariance is not yet
wired into every optical Thomson or Full-EM CTS fit.

## CTS reference-PCA basis uncertainty

The electrostatic CTS inverse bootstraps the supplied calibrated reference
spectra and returns covariance for the reference mean and sign/rotation-aligned
PCA basis, plus singular-value gap diagnostics. Rank-deficient bootstrap
subspaces are discarded.

Important scope: this estimates finite-reference-sample uncertainty of the
PCA basis. It is **not yet propagated into the fitted plasma-parameter
covariance**. The fitted nuisance-coefficient covariance remains conditional
on the selected basis. The result reports this distinction explicitly.

## Full-EM thermal background: free-free and ECE

`cts_em_background` computes non-relativistic Maxwellian thermal
Bremsstrahlung emissivity and integrates the LTE radiative-transfer equation.
ECE is represented using Kirchhoff's law with a supplied mode-resolved
absorption coefficient from a magnetized dielectric/radiative-transfer solver.
The solver does not fabricate an ECE absorption coefficient from only
`B`, `n_e` and `T_e`.

The output is spectral radiance. Use
`radiance_to_receiver_spectral_power` to convert it to receiver-plane spectral
power using measured receiver etendue and end-to-end efficiency exactly once.
`analyze_full_em_cts_spectrum` accepts a fixed additive
`physical_background_spectrum_W_Hz` already expressed at the same receiver
plane and in the same spectral units as the measured spectrum.

Free-free emissivity is non-relativistic and unmagnetized. Use a relativistic,
magnetized emissivity model outside that regime. The mode-resolved ECE
absorption coefficient must come from the appropriate magnetized plasma
calculation.

## Ray tracing and paraxial beam covariance

`trace_cold_plasma_ray` now supports two propagation paths. The 4-D transverse
path accepts a covariance plus one supplied symplectic 4x4 ABCD map per adjacent
ray sample. The 6-D path accepts an initial covariance in
`[x,y,z,kx,ky,kz]` and derives the local linearized Hamiltonian transfer from
the Hessian of the selected cold-plasma dispersion eigenvalue along the ray.
The 6-D path rejects propagation across an eigenvalue-order branch crossing.
`trace_cts_frequency_resolved_beams` uses the propagated 2-D transverse
covariances in the local Gaussian overlap integral. The transverse overlap is
two-dimensional; its units and path-integrated units are returned explicitly.

The 6-D method is a paraxial Gaussian phase-space approximation based on the
local dispersion Hessian, not a complete full-vector quasi-optical solver.
Non-paraxial diffraction, general vector mode conversion, absorption and
non-Gaussian beam evolution are not implied by this interface.

## Full-EM relativistic electron and fast-ion VDF shape inversion

`fit_full_em_gyrotropic_population_shape` performs a nonlinear,
regularized inversion of a tabulated gyrotropic momentum-space distribution.
Each optimizer evaluation calls the Full-EM relativistic kinetic forward model,
including the conductivity, Klimontovich fluctuation covariance and
second-order kinetic vertex. This is not a linear superposition of
single-particle kernels.

Two population modes are available:

- `population_kind="electron"`: infer the relativistic gyrotropic electron
  distribution shape.
- `population_kind="fast_ion"`: infer the relativistic gyrotropic fast-ion
  distribution shape.

The inversion is conditional on fixed bulk `n_e`, `T_e`, thermal-ion
temperatures, magnetic field, geometry, mode polarizations, instrument
response and population density. The electron branch requires an explicitly
zero projected electron flow because a drifted relativistic gyrotropic VDF is
not implemented. Identifiability is assessed from the data Jacobian, not from
regularization rows. A rank-deficient fit returns a conditional regularized
candidate rather than claiming a uniquely measured distribution.

## Multi-view 3-D EVDF reconstruction

`reconstruct_multiview_3d_evdf` solves a non-negative, regularized 3-D
velocity-space inverse problem from stacked view/frequency kernels. For a
relativistic Thomson result, each view's kernel must be generated using the
relativistic Doppler mapping, polarization and differential cross-section
appropriate to that view. A single view or rank-deficient stacked operator
cannot identify an unrestricted 3-D EVDF; the routine returns only a
conditional candidate in that case.

The current API consumes per-view spectral kernels. Automatic generation of
those kernels from arbitrary experimental geometry and a fully relativistic
3-D optical Thomson forward solver remains separate work.

## Completion and limitations

The code now contains working API-level implementations for the count
likelihood, independent CTS discrepancy covariance, free-free/ECE background
transfer, paraxial covariance propagation, nonlinear Full-EM gyrotropic shape
inversion and multi-view 3-D EVDF inversion. The following uncertainty and
integration boundaries remain explicit:

1. CTS PCA basis covariance is estimated but not propagated into plasma
   parameter covariance.
2. Optical Thomson Poisson likelihood is a separate count-space Maxwellian
   API; it is not a likelihood option for Kappa or relativistic fits.
3. The GP discrepancy covariance is consumed by electrostatic CTS, not every
   optical and Full-EM inverse.
4. Full-EM ECE requires externally supplied mode-resolved absorption.
5. The 6-D ray covariance path derives a linearized paraxial transfer from the
   selected cold-plasma dispersion Hessian; the alternative 4-D path accepts
   supplied ABCD maps. Full vector quasi-optical mode conversion and
   non-paraxial diffraction are not solved.
6. Drifted relativistic electron VDF inversion is not implemented.
7. 3-D EVDF kernels must be supplied per view and must themselves contain the
   appropriate relativistic scattering physics.
8. The new APIs are registered in the diagnostic registry, but not all have
   dedicated dashboard controls or automated orchestration paths.

## Scientific references

- Rybicki, G. B. & Lightman, A. P., *Radiative Processes in Astrophysics*
  (1979): thermal free-free emissivity and radiative-transfer conventions.
- Bekefi, G., *Radiation Processes in Plasmas* (1966): magnetized plasma
  emission and absorption framework.
- Verdier et al. (2025), *Review of Scientific Instruments* 96, 043513,
  DOI: 10.1063/5.0238948: CTS thermal/fast-ion background estimation.
- Carole et al. (2025), *Journal of Instrumentation* 20, P07014,
  DOI: 10.1088/1748-0221/20/07/P07014: high-resolution Thomson spectral
  modelling for WEST.
- Skolar, Longley & Goodwin (2025), *Physics of Plasmas*,
  DOI: 10.1063/5.0261966: generalized plasma-dispersion treatment for
  non-Maxwellian Thomson scattering.
- Sakai et al. (2026), *Journal of Instrumentation* 21, C01002,
  DOI: 10.1088/1748-0221/21/01/C01002: Monte Carlo Thomson spectra from
  arbitrary electron distributions.
- Flocken et al. (2026), *Fusion Engineering and Design*,
  DOI: 10.1016/j.fusengdes.2026.115845: accuracy-driven CTS optimisation,
  ray tracing and Bayesian inversion.

No experimental validation has been performed.
