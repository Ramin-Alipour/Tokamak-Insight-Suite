# Landau--FDT CTS and kinetic ECE integration

This implementation adds two orchestration layers to the existing numerical
kernels on main.

## Landau--FDT electrostatic CTS

The function landau_cts_dynamic_structure_factor in tokamak_analysis/cts_landau_fdt.py
constructs species density-observation and potential-energy forcing operators
from the assembled multispecies linearized Landau system. It applies the
multispecies Poisson interaction matrix V_ab=q_a*q_b/(epsilon_0*k^2) to the
bare response, screens the matching velocity-space Langevin spectrum with
the same dielectric resolvent, and independently evaluates classical matrix
FDT. The returned consistency residual compares the screened Langevin
spectrum with the FDT spectrum.

The solver is restricted to a common-temperature/common-flow isotropic
Maxwellian mixture, finite 3-D velocity box, non-relativistic kinetics and
electrostatic longitudinal fluctuations. It is not a magnetized Full-EM
three-wave scattering vertex and does not supply absolute optical power
normalization. Velocity-grid/domain convergence remains mandatory.

## Polarized kinetic ECE

solve_two_mode_kinetic_ece_profile in tokamak_analysis/ece_kinetic_modes.py
connects local magnetized Maxwellian conductivity to a two-mode opacity matrix
and exact ordered piecewise-constant coherency transfer. It retains
cross-polarization absorption and coherent mode coupling. The caller must
supply local wavevectors, magnetic fields, temperatures, densities and a
two-column unit-flux eigenmode basis for each segment/frequency.

The implementation remains non-relativistic and LTE. It does not infer ray
geometry or eigenmodes, and it does not replace the Berreman boundary solver.

## Vector propagation and mode conversion

The existing propagate_vectorial_angular_spectrum is an exact non-paraxial
vector propagator for homogeneous isotropic segments. The existing oblique
Berreman solver and solve_stratified_mode_conversion provide coherent
reflection/transmission Jones matrices for a one-dimensionally stratified
anisotropic dielectric, including off-diagonal mode conversion. They do not
solve arbitrary 3-D inhomogeneous anisotropic interfaces or transverse
diffraction through such interfaces.

## Verification boundary

Unit tests cover the new Landau/Poisson/FDT and two-mode ECE transfer
orchestration. CI must pass on the resulting main commit. Numerical tests
are not experimental validation; quantitative use requires velocity-grid,
harmonic, quadrature, spatial-step and boundary convergence studies.
