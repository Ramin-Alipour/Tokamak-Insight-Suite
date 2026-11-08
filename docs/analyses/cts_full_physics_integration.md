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


## Static screening and inference outputs for electron mixtures

For co-moving electron populations, the zero-frequency longitudinal
susceptibility gives the effective screening length

lambda_De,eff^-2 = (n_e e / epsilon_0) sum_j eta_j g_j/T_j,

where g_j=1 for a Maxwellian component and
g_j=(2 kappa_j - 1)/(2 kappa_j - 3) for a finite-variance Kappa component. The
scattering parameter is alpha=1/(k lambda_De,eff). This reduces to the
Maxwellian Debye length for a single Maxwellian and includes the exact
zero-frequency Kappa susceptibility enhancement for the stated 1-D
isotropic-Kappa marginal.

For counter-streaming electron populations there is no common electron rest
frame in which a scalar static Debye length describes the mixture. The API
therefore returns null for the scalar Debye length and alpha, together with an
explicit status, rather than reusing the reference temperature in a misleading
Maxwellian expression. The inverse also distinguishes the fitted reference
temperature from the total mixture second-moment temperature and the
density-weighted bulk electron flow.


## Tabulated-EVDF Landau susceptibility derivative

For the piecewise-linear projected VDF with its explicit zero extension, define
I_p(v_phi)=integral_L f'(v)/(v-v_phi-i eta)^p dv. The longitudinal
susceptibility is chi_e=-(omega_pe^2/k^2) I_1. Its phase-velocity derivative
at fixed positive retarded displacement is
d chi_e/d v_phi=-(omega_pe^2/k^2) I_2, and the angular-frequency derivative is
d chi_e/d omega=(1/k) d chi_e/d v_phi. The public CTS physics dispatcher and
Dashboard expose this optional analytic derivative. It is computed by the
second-order Landau kernel, not by finite differences; the supplied finite
support and endpoint-jump convention remain explicit.


The third central moment of a 1-D Kappa marginal is finite only for
kappa > 2. Therefore mixture skewness is reported only when every Kappa
component has kappa > 2. The variance threshold remains kappa > 3/2, and the
fourth-moment/kurtosis threshold remains kappa > 5/2; these are distinct
existence conditions and are enforced independently.


## Maxwell propagation of Landau–FDT current fluctuations

The new public model `landau_fdt_full_em_field_fluctuations` propagates the
current-noise tensor produced by `landau_fdt_full_em_constitutive` through the
local Maxwell Green dyadic. With the exp(-i omega t) convention,

```
D(k,omega) = k^2 I - k k^T - (omega/c)^2 epsilon_r
G = inverse(D)
S_E = mu_0^2 omega^2 G S_J G^dagger
```

The solver preserves Hermitian symmetry and checks positive semidefiniteness
of the propagated field spectrum. It rejects ill-conditioned undamped
Maxwell poles instead of returning arbitrarily large finite values. The
advanced Analysis-1 dispatcher, shared Dashboard selector and formula registry
expose this physical path. It remains a local homogeneous-medium field
fluctuation calculation, not a magnetized CTS three-wave scattering vertex,
ray solution or absolute receiver spectrum.


## Maxwell propagation of Landau–FDT fluctuations

The public model `landau_fdt_full_em_field_fluctuations` propagates the
current-noise tensor from `landau_fdt_full_em_constitutive` through the local
Maxwell Green dyadic. With the exp(-i omega t) convention:

```
D = k^2 I - k k^T - (omega/c)^2 epsilon_r
G = inverse(D)
S_E = mu_0^2 omega^2 G S_J G^dagger
S_B = (k_cross/omega) S_E (k_cross/omega)^dagger
B = k x E / omega
Pi = (1/(2 mu_0)) Re <E x B*>_omega
```

The electric and magnetic spectral tensors retain Hermitian positive
semidefiniteness. The Poynting-flux vector is derived from the E-B cross
spectrum. Ill-conditioned undamped Maxwell poles are rejected. The API,
Dashboard selector and both deployed/source formula registries expose this
route. It remains a local homogeneous-medium fluctuation calculation, not a
magnetized CTS three-wave scattering vertex, ray solution or absolute
receiver spectrum.


## Transverse polarization-resolved Full-EM fluctuation observables

Before Maxwell propagation, the Landau--FDT current spectral matrix is
required to be Hermitian and positive semidefinite. Invalid covariance is
rejected rather than silently symmetrized.

For a deterministic right-handed transverse analyzer basis e1,e2 perpendicular
to k, the projected electric coherency matrix is
C_ab(omega) = e_a^dagger S_E(omega) e_b.
The exported Stokes spectral densities use
(I,Q,U,V) = (C11+C22, C11-C22, 2 Re(C12), 2 Im(C12))
for the solver's exp(-i omega t) convention. The degree of polarization is
sqrt(Q^2+U^2+V^2)/I, defined as zero when I=0; positivity requires it not to
exceed one beyond numerical roundoff. These are local polarization-resolved
spectral observables for downstream receiver modeling. They do not supply
receiver etendue, absolute calibration, ray tracing, or a magnetized CTS
three-wave vertex.


## Relativistic optical Thomson momentum quadrature

The isotropic Maxwell--Juttner radial integral is evaluated after changing
variables from dimensionless momentum u to dimensionless kinetic energy
t=(sqrt(1+u^2)-1)/theta. With
u=sqrt(theta*t*(theta*t+2)), the radial measure becomes
u^2 du = theta*u*(theta*t+1) dt. Generalized Gauss--Laguerre quadrature with
weight exp(-t)*sqrt(t) therefore integrates the complete semi-infinite
momentum domain, including the relativistic tail, without a finite momentum
cutoff or renormalization of omitted particles. The numerical partition
integral is checked against theta*K_2(1/theta), evaluated with the scaled
modified Bessel function. Angular and gyrophase convergence remain separate
requirements; this change does not remove the wavelength-window capture gate
or establish experimental validation.
