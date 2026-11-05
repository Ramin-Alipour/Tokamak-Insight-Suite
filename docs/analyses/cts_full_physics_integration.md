# Integrated Landau--FDT, vector Maxwell and kinetic ECE execution

The `tokamak_analysis.cts_full_physics` module provides public orchestration
entry points for the advanced physics kernels:

- `kappa_electrostatic_susceptibility`: exposes the collisionless, unmagnetized longitudinal susceptibility of the exact 1-D marginal of an isotropic 3-D Kappa distribution. This is a response kernel only; it does not supply a Kappa fluctuation/noise closure or a complete CTS spectrum.
- `analyze_landau_fdt_cts`: assembles the conservative multispecies
  linearized Landau operator, derives the matching classical Langevin source
  covariance from the same discrete operator, and connects the screened
  electrostatic density response to the CTS dynamic structure factor.
- `propagate_stratified_nonparaxial_vector_field`: applies a 2-D angular
  spectrum to a z-stratified anisotropic dielectric. Each propagating transverse
  Fourier component is solved by the oblique full-vector Berreman boundary
  problem. The result retains diffraction, longitudinal electric field,
  reflection and coherent Jones mode conversion without a paraxial expansion.
- `solve_polarized_kinetic_ece_with_mode_conversion`: returns the
  two-mode magnetized kinetic ECE opacity/ordered LTE transfer together with
  the independent coherent Berreman reflection/transmission matrices for each
  frequency. A boundary Jones matrix is not misrepresented as a distributed
  coupling generator; the latter remains an explicit input to the transfer.

## Physics and numerical boundaries

These are substantive integration layers, not claims that every plasma regime
has been solved:

1. The assembled Landau reference operator is non-relativistic and assumes an
   isotropic Maxwellian mixture with common temperature and bulk flow. Its
   finite velocity box, grid, frequency and species limits require convergence
   studies. It is electrostatic longitudinal CTS; magnetized Full-EM scattering
   still needs a consistent electromagnetic current vertex and coupling.
2. The beam propagator is exact in transverse wave number for a medium
   invariant in x/y and stratified in z. Vacuum ports must propagate. Evanescent
   angular components are explicitly discarded. Arbitrary 3-D interfaces,
   lateral inhomogeneity, and evanescent-port matching are not claimed.
3. Kinetic ECE uses a non-relativistic Maxwellian Vlasov conductivity and LTE
   Rayleigh--Jeans transfer. The local wavevectors, unit-normal-flux modal basis,
   and distributed mode-coupling generator must be supplied consistently with
   the dielectric. The returned Berreman Jones matrices describe coherent
   boundary conversion and are reported separately from distributed opacity.
4. Relativistic/collisional ECE emissivity, arbitrary non-Maxwellian kinetic
   equilibria, fully coupled 3-D ray/eigenmode evolution, and experimental
   validation remain outside this integration layer.

Unit tests verify software contracts and limiting cases. They do not establish
experimental accuracy. A physical result must not be released quantitatively
until velocity-space, harmonic, quadrature, spatial-step, angular-spectrum and
boundary convergence have been demonstrated for the supplied regime.


## Electron-mixture moment closure for collective Kappa CTS

The collective kinetic spectrum sums all supplied electron-population
susceptibilities before constructing the common dielectric response.
For density fractions eta_j, projected drifts u_j and component second-moment
temperatures T_j, the physical bulk flow is u_bar = sum_j eta_j u_j.
The total projected second-moment temperature is
T_e,2 = sum_j eta_j T_j + (m_e/e) sum_j eta_j (u_j-u_bar)^2.
The second term is the between-population streaming contribution; omitting it
reports only the weighted component temperature, not the variance of the
total electron distribution.

For symmetric Maxwellian/Kappa components, the third central moment is
sum_j eta_j [3 delta_j sigma_j^2 + delta_j^3]. A 1-D Kappa marginal has a
finite fourth central moment only for kappa > 5/2, with
mu4_j = 3 (2 kappa_j - 3)/(2 kappa_j - 5) sigma_j^4. The mixture fourth
moment adds the exact shift terms 6 delta_j^2 sigma_j^2 + delta_j^4.
When any nonzero Kappa component has 3/2 < kappa <= 5/2, variance remains
finite but kurtosis diverges and is returned as unavailable, never clipped.
The public advanced-physics API and its Dashboard example expose these
derived moments alongside the summed multi-population dielectric.
