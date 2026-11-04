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


## Relativistic projected optical Thomson inversion: Compton completion

The public advanced dispatcher route
`thomson_relativistic_projected_evdf` now supports two explicit regimes:

- The default relativistic Thomson-limit inverse retains exact Doppler
  kinematics and screens the maximum rest-frame recoil correction against
  `maximum_recoil_correction_fraction` (default 0.01). It refuses to proceed
  when the selected no-recoil approximation exceeds that error budget.
- With `include_compton_recoil=true`, the inverse uses exact single-electron
  four-momentum Compton kinematics, Lorentz aberration into the electron rest
  frame, and the polarized Klein–Nishina differential cross-section. The
  researcher must supply the incident polarization vector, transverse to the
  incident ray in the declared symmetric geometry.

The velocity distribution remains a one-dimensional marginal along the
scattering vector. This does not recover transverse velocity structure or an
unrestricted 3-D EVDF, and it does not infer absolute density. The reported
cross-section weight is relative to the classical electron-radius-squared
scale; absolute density still requires the independent calibrated-power
geometry and efficiency contract. The forward and inverse recoil switches are
separate APIs and must be selected consistently by the caller.

Regression tests cover the stationary-electron Compton wavelength shift,
polarization-dependent Klein–Nishina weighting, the no-recoil validity screen,
and the required polarization input. These tests establish numerical software
contracts only; they are not experimental validation.
