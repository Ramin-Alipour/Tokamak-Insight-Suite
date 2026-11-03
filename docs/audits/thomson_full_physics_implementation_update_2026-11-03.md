# Thomson analysis: implementation update (2026-11-03)

This change adds two executable numerical components to the main branch.

## Added

- `tokamak_analysis/cts_full_wave.py`: 1-D stratified, full-vector Maxwell propagation using a Berreman 4x4 tangential-field system. It retains coherent polarization coupling and solves the slab boundary matching problem. It is not a general 3-D Maxwell solver and does not include transverse diffraction.
- `tokamak_analysis/ece_kinetic.py`: mode-polarization-resolved absorption derived from the existing magnetized Maxwellian Vlasov conductivity in `cts_full_em.py`, using Poynting's theorem. It is not relativistic/Juttner ECE and requires a supplied local wavevector and polarization.

## Still not complete; do not represent as implemented

- TS-01: the existing conserving BGK kernel is not a full multispecies linearized Landau collision operator. A physically complete implementation must include test-particle and field-particle terms, conservation of number/momentum/energy, and a fluctuation source consistent with the same operator and FDT. This commit does not relabel BGK as Landau-FDT.
- TS-02: kinetic Maxwellian ECE absorption is now available for a supplied local mode. A fully relativistic, distribution-averaged ECE emissivity/absorption and self-consistent transfer/ray coupling remains open.
- TS-03: relativistic and magnetized bremsstrahlung corrections remain open.
- TS-04: the new solver is full-vector only for a 1-D stratified slab at normal incidence; arbitrary 3-D non-paraxial propagation and diffraction remain open.
- TS-05: coherent mode conversion is retained by the coupled 1-D Maxwell system. General 3-D conversion at cutoffs/resonant layers and reflection remains open.
- TS-06: the existing EVDF paths do not yet constitute a general relativistic anisotropic distribution closure across every CTS/Optical route.
- TS-07–TS-20: require separate cross-route integration and CI verification; this commit does not assert blanket completion.

## Numerical validation

Added tests cover vacuum propagation (zero reflection/unit transmission), singular dielectric rejection, Berreman matrix shape/finiteness, and finite kinetic absorption output. The tests are committed but have not been executed in the GitHub Actions environment at commit creation time. CI status must be read from the commit's Actions run.

## Conventions

The full-wave solver uses exp(-i omega t), vacuum exterior regions, normally incident fields, and SI units. The ECE absorption routine reports alpha in m^-1, positive for net dissipation under the same time convention. No empirical calibration or experimental validation is implied.
