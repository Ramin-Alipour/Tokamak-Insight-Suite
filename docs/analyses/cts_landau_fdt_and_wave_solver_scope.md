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
