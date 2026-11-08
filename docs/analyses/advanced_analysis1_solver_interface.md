# Advanced Analysis-1 solver interface

The Advanced Analysis workspace now exposes the four explicit numerical
entrypoints in `tokamak_analysis.cts_full_physics.run_advanced_thomson_physics`:

- `landau_fdt_cts`: multispecies finite-box linearized Landau operator, matching
  classical Langevin/FDT covariance and self-consistent electrostatic Poisson
  screening. Restricted to common-temperature/common-flow isotropic Maxwellians.
- `polarized_kinetic_ece`: magnetized non-relativistic Maxwellian kinetic
  opacity, two-mode coherency transfer and a separately solved stratified
  Berreman/Jones boundary conversion.
- `stratified_nonparaxial_vector`: non-paraxial transverse angular spectrum
  coupled to an oblique full-vector Berreman solve for a dielectric stratified
  only along z.
- `stratified_mode_conversion`: coherent oblique-incidence reflection and
  transmission Jones matrices for a supplied one-dimensional stratified
  dielectric tensor.

The Streamlit page accepts a JSON object of explicit solver keyword arguments.
Complex arrays are represented by an object with exactly `real` and `imag`
arrays of equal shape. It does not infer geometry, dielectric tensors, mode
bases, plasma profiles, calibration or uncertainties from an arbitrary upload.

## Scientific scope and release gates

These are concrete numerical kernels, not claims of universal full-physics
closure. The Landau reference operator requires velocity-grid/domain and
frequency convergence and remains non-relativistic, electrostatic and
Maxwellian. Kinetic ECE remains non-relativistic Maxwellian/LTE; the polarized
opacity transfer and Berreman boundary conversion are returned as distinct
physical operators, not falsely represented as one fully coupled 3-D transfer
equation. Vector propagation and mode conversion are exact only within the
z-stratified, transversely invariant dielectric model; evanescent angular
components are not matched through the slab. General 3-D inhomogeneous
interfaces, relativistic/collisional ECE, arbitrary non-Maxwellian Landau
equilibria and experimental validation remain outside this interface.

The page preserves these restrictions in solver output and exports them with
the numerical results. Successful numerical execution is not a measurement
validation or a substitute for convergence studies.
