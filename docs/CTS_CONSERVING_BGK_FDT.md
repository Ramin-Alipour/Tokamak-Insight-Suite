# Conserving-BGK electrostatic CTS closure

## Implemented physical chain

The Analysis-1 advanced physics API now exposes
`cts_conserving_bgk_fdt` through
`tokamak_analysis.cts_full_physics.run_advanced_thomson_physics`.

For each species the kinetic solver computes the isothermal,
particle-number-conserving BGK longitudinal response with the common
momentum-restoring velocity. In the `exp(i k x - i omega t)` convention its
linearized operator is

\[
C_s=-\nu_s\left[\delta f_s-f_{0s}
\left(\frac{\delta n_s}{n_s}
+\frac{m_s(v-u)\delta u}{T}\right)\right],
\]

with the common velocity perturbation fixed by total mixture momentum
conservation. The returned bare response is
\(R_{ab}=\delta n_a/\delta U_b\), in `m^-3/J`.

Self-consistent electrostatic screening is then applied using

\[
V_{ab}(k)=\frac{q_a q_b}{\epsilon_0 k^2},\qquad
\epsilon(k,\omega)=I-R(k,\omega)V(k),\qquad
\chi=\epsilon^{-1}R.
\]

The classical matrix fluctuation-dissipation closure uses the retarded
response and the `exp(-i omega t)` convention:

\[
S(k,\omega)=-\frac{2T}{\omega}
\operatorname{Im}_{H}\chi(k,\omega),\qquad
\operatorname{Im}_{H}\chi=\frac{\chi-\chi^\dagger}{2i}.
\]

The FDT implementation rejects non-positive-semidefinite spectra beyond the
configured numerical tolerance. Zero frequency is rejected; a static value
must come from an independently converged low-frequency extrapolation.

## API

```python
from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

result = run_advanced_thomson_physics(
    "cts_conserving_bgk_fdt",
    angular_frequency_rad_s=omega,
    wave_number_m_inverse=k,
    species=[
        {"name": "electron", "density_m3": ne, "mass_kg": me, "charge_C": -e},
        {"name": "deuterium", "density_m3": ni, "mass_kg": md, "charge_C": e},
    ],
    common_temperature_J=T,
    collision_frequencies_s_inverse=[nu_e, nu_i],
    quadrature_order=96,
    quadrature_max_order=384,
    quadrature_convergence_tolerance=1e-5,
)
```

The result includes the bare and screened response matrices, dielectric
matrix and condition number, Coulomb interaction matrix, dynamic structure
factor, collision regime, and explicit validity/convergence gates. The BGK
velocity integral is evaluated at the requested Hermite order and then
re-evaluated at successively doubled orders (up to `quadrature_max_order`).
The relative Frobenius-norm change of the complete multispecies response
matrix is reported as `velocity_quadrature_convergence.relative_response_change`.
A quantitative-release flag is true only when this change meets
`quadrature_convergence_tolerance`; reaching the maximum order without meeting
the tolerance leaves the result explicitly non-converged.

## Scope boundary

This is a useful collisional electrostatic CTS branch. It is not a replacement
for the discrete Coulomb Landau-FDT solver, and it is not the magnetized
Full-EM three-wave scattering vertex. It assumes a common species temperature
and common equilibrium flow, conserves each species particle number and total
mixture momentum, but does not conserve total energy. The classical FDT
closure does not add quantum detailed balance. Velocity-quadrature,
frequency-resolution and collision-model convergence are required, and no
experimental validation is claimed.

The model is available through the public Analysis-1 advanced physics
dispatcher. It must not be labelled as a completed Full-EM CTS workflow or as
absolute receiver power without the separate calibrated scattering and
receiver transfer chain.


## Numerical invariance checks

The BGK kernel test suite also checks the Galilean frequency shift
\(\omega' = \omega-kU\) for a common equilibrium drift and covariance of
the response matrix under a simultaneous permutation of species densities,
masses and collision frequencies. These are independent physical invariance
checks; they complement the static Boltzmann limit and matrix-screening/FDT
closure tests.
