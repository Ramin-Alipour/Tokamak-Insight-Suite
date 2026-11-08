# Non-paraxial vector propagation: implementation boundary

The new `tokamak_analysis.cts_nonparaxial.propagate_nonparaxial_vector_field`
implements exact vector angular-spectrum propagation for a homogeneous,
isotropic, lossless medium. It uses the full longitudinal wave number and
projects each Fourier component onto the Maxwell-transverse subspace. It does
not use a paraxial expansion.

This is deliberately **not** described as the requested complete magnetized
plasma Full-EM propagation. In particular, it does not solve an inhomogeneous
tensor-dielectric Maxwell boundary-value problem or resonant mode-conversion
layers. The existing `cts_em_geometry.propagate_cold_plasma_mode_coupling`
is a local lossless coupled-mode model, not a substitute for those full-wave
calculations.

The complete Landau-FDT collision operator and the kinetic ECE absorption
solver remain separate physics tasks. Existing Spitzer rates and BGK response
must not be relabelled as Landau-FDT, and ECE brightness-temperature inversion
must not be relabelled as a kinetic absorption/emission solution.
