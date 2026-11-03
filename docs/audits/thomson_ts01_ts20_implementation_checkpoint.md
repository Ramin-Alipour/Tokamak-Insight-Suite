# Thomson advanced-physics implementation checkpoint (TS-01–TS-20)

This branch adds three independently callable numerical kernels. It does NOT claim that all twenty audit items are complete.

| ID | Status in this branch | Scope / remaining work |
|---|---|---|
| TS-01 | Partial | 3-D Landau test-particle Fokker–Planck operator with Rosenbluth convolution. A conservative multispecies field-particle discretization and full collective Boltzmann–Langevin/FDT covariance are still required. |
| TS-02 | Partial | EVDF-integrated ECE emissivity/absorption and formal transfer for caller-supplied relativistic harmonic kernels. A validated built-in polarized cyclotron/synchrotron kernel and independent kernel benchmarks remain required. |
| TS-03 | Open | No complete magnetized relativistic bremsstrahlung kernel is added here. |
| TS-04 | Partial | Full-vector non-paraxial 1-D stratified Maxwell ODE for anisotropic dielectric tensors. 2-D/3-D geometry, boundary matching, reflections and scattering matrix remain open. |
| TS-05 | Open | Full mode-conversion coefficients require two-sided boundary matching and branch tracking; not represented as complete by the IVP propagator. |
| TS-06 | Open | General relativistic anisotropic EVDF fitting and normalization/identifiability validation remain open. |
| TS-07–TS-20 | Not certified | Must be reconciled against every Optical/CTS call path, dashboards, tests and documentation; this branch does not mark them complete. |

## Numerical contracts

* cts_landau_fdt.landau_test_particle_fokker_planck accepts a non-negative 3-D Cartesian velocity distribution and uniform velocity axes. The local Ito covariance returned is 2 D; it is explicitly the test-particle diffusion covariance, NOT the complete collective FDT noise operator.
* ece_kinetic.kinetic_ece_coefficients requires emissivity and absorption kernels whose units, harmonic/polarization convention, relativistic resonance and induced-emission derivative are defined by the caller. The integration and formal transfer are implemented; a kernel is not silently fabricated.
* cts_full_vector_em.propagate_full_vector_em integrates Maxwell's equations without a paraxial approximation for a z-stratified 3x3 dielectric tensor. It is an initial-value propagator, not a complete reflection/mode-conversion boundary solver.

These APIs are intentionally explicit about their physical validity domains. Do not promote this checkpoint to a validated production model without analytic-limit tests, convergence studies, conservation/FDT tests and comparison against independent published benchmarks.
