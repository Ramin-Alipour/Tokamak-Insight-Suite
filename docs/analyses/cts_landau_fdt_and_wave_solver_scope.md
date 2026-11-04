# CTS collision/FDT and electromagnetic propagation implementation boundary

This note records the verified implementation scope. It must not be read as
claiming that the existing modules form a complete Landau-FDT solver.

## Implemented components on main

- cts_collisional_bgk.py: isothermal multispecies BGK longitudinal response,
  conserving each species particle number and total mixture momentum. It does
  not conserve total energy and is not the Landau operator.
- cts_fdt.py: classical matrix fluctuation-dissipation conversion for a
  supplied retarded density susceptibility. It checks Hermitian positive
  semidefiniteness and requires an explicit zero-frequency slope. It is an
  FDT closure utility, not a velocity-space Landau stochastic source.
- ece_kinetic.py: magnetized, non-relativistic Maxwellian Vlasov conductivity
  absorption normalized by supplied mode normal Poynting flux, plus scalar
  LTE Rayleigh-Jeans slab transfer.
- cts_full_wave.py: full-vector oblique Berreman 4x4 propagation through a
  one-dimensionally stratified local dielectric, and coherent 2x2 Jones
  reflection/transmission matrices for polarization conversion.

## Still not complete

1. A conserving, velocity-dependent multispecies linearized Landau operator
   with test-particle and field-particle terms, energy conservation and
   verified discrete entropy production.
2. The matching velocity-space Langevin source covariance/FDT closure for
   that Landau operator, and its coupling into the CTS dynamic structure
   factor and Full-EM spectral dispatcher.
3. Relativistic and collisional ECE emissivity/absorption, and self-consistent
   ray/field-profile coupling of ECE transfer to the vector mode solver.
4. General 3-D non-paraxial diffraction and mode conversion at arbitrary
   interfaces. The Berreman solver is exact only for a 1-D stratified medium
   with conserved transverse wavevector and propagating vacuum ports.

Unit tests establish software behavior, not experimental or device validation.


## New main-branch implementation

- `cts_landau_fdt.py`: assembled finite-box 3-D Cartesian-velocity multispecies
  linearized Coulomb Landau operator in symmetric weak form. Both test-particle
  and field-particle contributions are retained. The weak form is a Gram sum,
  with explicit species-number, total-momentum and total-energy invariants and
  non-positive entropy production. The classical Langevin covariance is built
  from the same discrete operator, not from an independently chosen collision
  width. A Fourier streaming term can be included in the density resolvent.
  This is a reference discretization about a common-temperature/common-flow
  isotropic Maxwellian mixture; it is non-relativistic and requires velocity
  grid/domain convergence. It is not yet a drop-in replacement for the
  electrostatic/Full-EM CTS dispatcher, and it does not supply self-consistent
  Poisson screening or optical Thomson normalization.
- `cts_full_wave.py`: `propagate_vectorial_angular_spectrum` performs exact
  non-paraxial vector angular-spectrum propagation through homogeneous
  isotropic segments, retaining evanescent components and reconstructing the
  longitudinal electric field from Maxwell transversality. Existing oblique
  Berreman propagation and the 2x2 Jones solver retain coherent polarization
  conversion for 1-D stratified anisotropic media.
- `ece_kinetic.py`: the independent magnetized Maxwellian kinetic absorption
  coefficient and ordered slab transfer are available. They remain
  non-relativistic and use caller-supplied local mode wavevector/polarization;
  the scalar profile transfer is not yet a fully coupled polarized 3-D
  radiative-transfer solution.

## Remaining integration boundary

The new Landau-FDT operator is a real velocity-space collision/FDT kernel, but
its CTS dielectric screening, measured-channel projection, Full-EM coupling,
and dashboard exposure still require explicit integration and regression
tests. The angular-spectrum propagator solves homogeneous segments only;
general 3-D anisotropic plasma interfaces and diffraction remain outside its
scope. Mode conversion is coherent for the existing 1-D stratified Berreman
boundary problem, not arbitrary 3-D interfaces. ECE absorption remains
non-relativistic Maxwellian; relativistic/collisional emissivity and
mode-coupled 3-D transfer remain open. Do not mark these broader capabilities
as complete from the existence of these kernels alone.
