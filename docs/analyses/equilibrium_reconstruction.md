# Equilibrium reconstruction: Grad–Shafranov and magnetic inverse fit

## Scope

The workspace exposes two computational paths. They are complementary, but neither should be described as a full machine-specific EFIT implementation.

1. **Prescribed-boundary Grad–Shafranov solve** — solves the axisymmetric elliptic equation on a uniform R–Z grid for a supplied perimeter flux and tabulated p'(psi_N), F F'(psi_N) profiles. It returns psi, normalized flux, B_R, B_Z, B_p, iteration diagnostics, and an interior PDE residual.
2. **Weighted inverse flux-basis fit** — estimates coefficients of a user-supplied psi(R,Z) basis from calibrated magnetic-probe and/or flux-loop measurements. Optional coil response matrices and measured coil currents are subtracted before fitting. Measurement uncertainty is used as a per-observation weight; Tikhonov regularization is available.

## Prescribed-boundary inputs

Upload six CSV files in the Equilibrium & Reconstruction workspace:

- R_m: one-dimensional, positive, strictly increasing, uniformly spaced major-radius coordinates (m).
- Z_m: one-dimensional, strictly increasing, uniformly spaced vertical coordinates (m).
- boundary_flux_Wb_rad: 2-D array shaped (len(Z_m), len(R_m)); the perimeter is fixed.
- psi_profile: strictly increasing normalized-flux coordinates within [0, 1].
- pressure_gradient_Pa_per_Wb: p'(psi) values aligned with psi_profile.
- ffprime_T2: F F'(psi) values aligned with psi_profile.

The solver uses finite differences and successive over-relaxation. It reports convergence only if the iteration-change criterion and normalized interior equation residual both pass. Always inspect both. A failed or unconverged solve is not an equilibrium result.

The Grad–Shafranov convention used here is

`Delta* psi = -mu0 R^2 p'(psi) - F F'(psi)`

with `Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2`. The poloidal field convention is `B_R = -(1/R) dpsi/dZ`, `B_Z = (1/R) dpsi/dR`.

## Inverse-fit JSON bundle

The dashboard's inverse-fit uploader expects a JSON object with R_m, Z_m, basis_psi_Wb_rad (shape n_basis × n_Z × n_R), and at least one complete measurement block:

- Magnetic probes: magnetic_probe_response_T_per_coefficient, magnetic_probe_values_T, magnetic_probe_sigma_T.
- Flux loops: flux_loop_response_Wb_rad_per_coefficient, flux_loop_values_Wb_rad, flux_loop_sigma_Wb_rad.

Each response matrix has one column per basis coefficient. Each uncertainty is a positive 1-sigma value in the same units as the corresponding observation. Optional coil subtraction uses coil_probe_response_T_per_A with coil_probe_currents_A and/or coil_flux_response_Wb_rad_per_A with coil_flux_currents_A. Each coil response must align with the diagnostic's observation count and the current vector.

The weighted fit rejects a rank-deficient response matrix, since regularization cannot make unidentifiable coefficients identifiable. Outputs include basis coefficients, flux and normalized-flux grids, an interior-extremum axis estimate, a computational-edge flux contour estimate, per-diagnostic residuals, and the total weighted residual. The axis and edge contour are estimates from the supplied basis; they are not guaranteed to be the physical magnetic axis or last closed flux surface.

## Flux-map-derived equilibrium profiles

The public-data workflow now also estimates the Grad–Shafranov source profiles from the 2-D EFIT flux map. In each normalized-flux bin it fits

`Δ*ψ = -μ0 R² p'(ψ) - F(ψ)F'(ψ)`

jointly for `p'(ψ)` and `F F'(ψ)`, using the within-bin variation in `R²`. It integrates `p'` with the explicit boundary convention `p(edge)=0`. The output includes the magnetic-axis estimate, pressure-gradient profile, `F F'` profile, relative pressure profile, sample counts and design rank per flux bin.

This is an **inverse-source estimate**, not a unique kinetic-pressure reconstruction. A single flux map does not uniquely determine pressure and current profiles unless the joint source fit is identifiable; the UI reports the fit rank/identifiability and never treats the estimated profile as an independently measured profile. The public challenge dataset provides EFIT flux maps but withholds most profile labels on its test split.

The safety-factor profile `q(ψ)` is calculated only when the user supplies an independently known edge value `F(edge)=R Bφ` in T·m. The solver then integrates along closed flux contours that enclose the estimated axis. If the input is omitted, no `q` or `q95` is fabricated. The result remains sensitive to the flux-map sign convention, contour resolution, the supplied edge value, and the validity of the inferred source functions.

## Limitations and interpretation

- Supply machine-specific geometry, basis functions, response matrices, calibration, coil-current measurements, and uncertainties. The software does not generate them.
- The inverse fit is linear in basis coefficients; it does not perform a nonlinear free-boundary EFIT solve or jointly reconstruct pressure/current profiles.
- The prescribed-boundary solver does not infer the boundary or profiles from measurements.
- The generic module does not reconstruct q(r), fit X-point constraints, or propagate measurement/calibration/profile uncertainties.
- A successful numerical test demonstrates software behavior for its mathematical contract, not agreement with a particular device.
- Do not interpret an output as a physically complete equilibrium without checking identifiability, residuals, units, geometry, and the appropriateness of the supplied basis and profiles.

## Reproducible software checks

Run:

`python -m pytest -q tests/test_equilibrium_reconstruction.py tests/test_equilibrium_dashboard.py`

The dashboard tests exercise both missing-input handling and the two computational paths using a clearly identified synthetic acceptance fixture. The fixture is a software regression test, not experimental validation.
