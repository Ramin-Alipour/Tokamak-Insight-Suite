# Advanced CTS and optical Thomson inverse operators

This page records the scientific scope of the advanced operators added to
Analysis 1. All measured inputs remain researcher-prepared and calibrated.
No detector calibration, data repair, empirical validation or synthetic-to-
experimental equivalence is claimed.

## Optical Thomson: independent Poisson likelihood

The canonical optical Maxwellian fit accepts `likelihood="poisson"`.
It uses signed square-root Poisson-deviance residuals with expected counts
formed from the fitted spectral signal plus detector-background nuisance
terms. Inputs must be raw integer photon/count bins. Calibrated power-density
spectra must not be passed as counts.

The integrated Poisson path is currently restricted to
`non_collective_maxwellian`. Kappa, relativistic Maxwell-Jüttner and
collective optical fits retain their existing Gaussian/GLS likelihood paths.
Poisson counts cannot be combined with Gaussian pointwise uncertainties or
Gaussian spectral covariance in this interface.

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

The electrostatic CTS inverse now bootstraps the supplied calibrated reference
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

The output is spectral radiance. Convert it to receiver-plane spectral power
using the measured receiver etendue and end-to-end efficiency exactly once.
`analyze_full_em_cts_spectrum` accepts a fixed additive
`physical_background_spectrum_W_Hz` already expressed at the same receiver
plane and in the same spectral units as the measured spectrum. The radiative
transfer helper and the Full-EM inverse therefore have an explicit unit/plane
boundary.

Free-free emissivity is non-relativistic and unmagnetized. Use a relativistic,
magnetized emissivity model outside that regime.

## Ray tracing and paraxial beam covariance

`trace_cold_plasma_ray` can accept an initial transverse phase-space
covariance and one symplectic 4x4 paraxial transfer matrix per adjacent ray
sample. The covariance is propagated through the supplied Hamiltonian/ABCD
maps. `trace_cts_frequency_resolved_beams` uses the propagated 2-D transverse
covariances in the local Gaussian overlap integral.

This is a genuine covariance propagation path, but the transfer matrices are
supplied by the caller. The ray solver does not derive the full dispersion
Hessian or solve a general vector quasi-optical equation. Non-paraxial
diffraction, mode conversion, absorption and non-Gaussian beam evolution are
not implied by this interface.

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
not implemented. A rank-deficient fit returns a conditional regularized
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

The code now provides independent operators and explicit inverse paths for
the items above, but not every one is fully propagated through all dashboard,
registry and parameter-uncertainty surfaces. In particular:

1. CTS PCA basis covariance is estimated but not propagated into plasma
   parameter covariance.
2. Optical Thomson Poisson likelihood is currently integrated only for the
   non-collective Maxwellian model.
3. The GP discrepancy covariance is consumed by electrostatic CTS, not every
   optical and Full-EM inverse.
4. Full-EM ECE requires externally supplied mode-resolved absorption.
5. Paraxial transfer maps are supplied; the full anisotropic quasi-optical
   operator is not derived internally.
6. Drifted relativistic electron VDF inversion is not implemented.
7. 3-D EVDF kernels must be supplied per view and must themselves contain the
   appropriate relativistic scattering physics.
8. Dashboard and registry exposure must be synchronized before these
   functions are considered fully available through the graphical workflow.

No experimental validation has been performed.
