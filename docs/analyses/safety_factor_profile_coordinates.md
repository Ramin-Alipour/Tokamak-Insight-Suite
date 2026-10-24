# Safety-factor profile coordinate contract

The coordinate used for a safety-factor profile must be explicit and must match the requested physical quantity.

- `q_profile_psi_n`: normalized poloidal flux coordinate `psi_N`; interpolation at `psi_N = 0.95` is returned as `q95_provided`.
- `q_profile_rho`: normalized minor-radius coordinate `rho`; interpolation at `rho = 0.95` is returned as `q_at_rho_0_95`, not as conventional `q95`.

These coordinates are not generally interchangeable. In common tokamak convention, q95 is evaluated at the 95% normalized poloidal-flux surface. The conversion between normalized minor radius and normalized poloidal flux depends on the equilibrium and must not be assumed to be linear or identical.

Both coordinate arrays must be finite, strictly increasing, match the q-profile length, lie in [0, 1], and cover 0.95. If neither coordinate is supplied, the function deliberately does not infer q95 from sample indices. This interpolation is a profile operation, not a substitute for an equilibrium reconstruction or an independent validation of the q profile.
