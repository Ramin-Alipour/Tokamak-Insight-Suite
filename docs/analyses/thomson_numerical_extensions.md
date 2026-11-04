# Thomson / CTS numerical extensions

## Implemented numerical kernels

- `tokamak_analysis.cts_fdt.classical_fdt_dynamic_structure_factor` applies the
  classical matrix fluctuation--dissipation relation to an already-computed
  causal retarded susceptibility. It validates Hermitian positive
  semidefiniteness and refuses an uncomputed zero-frequency limit. This is a
  closure only; it does **not** compute a Landau collision operator.
- `tokamak_analysis.full_vector_maxwell.propagate_full_vector_stratified_maxwell`
  integrates the four tangential Maxwell fields for a z-stratified anisotropic
  dielectric tensor. It retains backward waves, reflection and polarization
  coupling without a paraxial approximation. It is not a 3-D finite-element
  solver and cannot cross epsilon_zz=0 in its present reduced formulation.
- `tokamak_analysis.ece_analysis.kinetic_ece_absorption` evaluates local
  magnetized nonrelativistic Maxwellian ECE absorption from the shared
  gyro-harmonic Vlasov conductivity. It returns alpha and an LTE Planck source
  coefficient. It does not solve ray-integrated radiative transfer or
  relativistic/nonthermal emissivity.

## Explicitly not claimed complete

The full multispecies linearized Landau collision operator (including
field-particle terms, exact conservation of particles/momentum/energy and
the associated stochastic/FDT source) is not implemented by these kernels.
Consequently, the FDT helper must not be described as a Landau--FDT CTS solver.
A full 3-D non-paraxial inhomogeneous wave solver, resonant-layer regularization,
and automatic boundary matching are also outside the stratified solver.
