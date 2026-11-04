# Thomson analysis: kinetic ECE transfer and full-vector mode conversion

This change adds two numerical interfaces: a piecewise-constant formal
Rayleigh-Jeans ECE transfer integrator, and coherent 2x2 reflected/transmitted
Jones matrices built by propagating both incident basis polarizations through
the oblique full-vector Berreman boundary-value solver.

Scope is deliberately explicit. Kinetic ECE absorption remains based on the
non-relativistic Maxwellian Vlasov conductivity and requires a supplied local
mode wavevector and polarization. The transfer step assumes LTE and a
Rayleigh-Jeans source function. The Berreman solver is exact for a local
dielectric tensor in a one-dimensionally stratified medium with conserved
transverse wavevector; it is not a general 3-D diffraction solver.

The CTS collision modules still provide collision-rate estimates and an
isothermal BGK response, not a conserving multispecies linearized Landau
operator. A Landau-consistent fluctuation source/FDT closure is not claimed.
That work requires the velocity-space test-particle and field-particle terms,
conservation identities, equilibrium covariance, and convergence tests to be
implemented and verified together.
