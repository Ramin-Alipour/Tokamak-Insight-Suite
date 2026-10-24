# Formula-library mathematical and physics audit

**Audit scope:** all 37 formula-registry entries mapped to `tokamak_analysis.formula_library`, plus shared input/integration helpers.  
**Audit date:** 2026-10-23  
**Audit status:** source-level review and targeted regression hardening; not a claim of complete independent empirical validation or a substitute for executing the full CI matrix.

## Interpretation and acceptance levels

- **Equation/units:** equation and declared input/output units are reviewed at source level against the named callable.
- **Numerical edge cases:** finite values, positive/non-negative domains, shape checks, singular denominators, and uncertainty-input pairing must be covered by implementation and tests.
- **Uncertainty:** propagated values are generally first-order independent-input standard uncertainties unless the function says otherwise. Correlations, calibration/systematic error, geometry error, model discrepancy, and instrument response are not automatically included.
- **Scientific status:** a numerically correct primitive is not automatically a validated diagnostic reconstruction. Empirical validation requires an independent compatible measurement/reference and reproducible provenance.

## Registry-wide formula inventory

| # | Callable | Equation | Inputs | Output | Declared assumptions / interpretation |
|---:|---|---|---|---|---|
| 1 | `tokamak_analysis.formula_library.electron_plasma_frequency` | ω_pe = sqrt(n_e e²/(ε₀ m_e)); f_pe=ω_pe/(2π) | n_e [m^-3] | rad/s, Hz | Local electron density; cold-plasma definition. |
| 2 | `tokamak_analysis.formula_library.debye_length` | λ_D = sqrt(ε₀ e T_e[eV]/(n_e e²)) | T_e [eV], n_e [m^-3] | m | Quasi-equilibrium Maxwellian electron approximation; temperature input is in eV. |
| 3 | `tokamak_analysis.formula_library.electron_cyclotron_frequency` | ω_ce=e\|B\|/m_e; f_ce=e\|B\|/(2πm_e) | B [T] | rad/s, Hz | Non-relativistic single-particle frequency; harmonics and relativistic shifts may matter. |
| 4 | `tokamak_analysis.formula_library.ion_sound_speed` | c_s≈sqrt(e(T_e+γ_i T_i)/m_i) | T_e,T_i [eV], m_i [kg], model γ_i | m/s | Model-dependent closure and ion species. |
| 5 | `tokamak_analysis.formula_library.electron_thermal_pressure` | p_e=n_e e T_e[eV] | n_e [m^-3], T_e [eV] | Pa | Temperature represented in electron-volts. |
| 6 | `tokamak_analysis.formula_library.ion_thermal_pressure` | p_i=Σ_s n_s e T_s[eV] | Species densities and temperatures | Pa | Sum over ion species; do not assume single-ion composition silently. |
| 7 | `tokamak_analysis.formula_library.total_thermal_pressure` | p_th=p_e+p_i | Species-consistent electron and ion pressures | Pa | Excludes energetic/non-thermal components unless included explicitly. |
| 8 | `tokamak_analysis.formula_library.thermal_stored_energy` | W_th=(3/2)∫p_th dV | Pressure profile and volume element | J | Classical isotropic ideal-gas form; geometry and species definitions matter. |
| 9 | `tokamak_analysis.formula_library.line_integrated_density` | N_e=∫ n_e dl | n_e along line of sight | m^-2 | A line integral; local profile needs multiple sightlines/model. |
| 10 | `tokamak_analysis.formula_library.plasma_frequency_from_density_cm3` | f_pe≈8.98×10^3 sqrt(n_e[cm^-3]) Hz | n_e [cm^-3] | Hz | Approximate electron plasma frequency; exact SI-constant implementation converts cm^-3 to m^-3. |
| 11 | `tokamak_analysis.formula_library.doppler_line_of_sight_velocity` | v_LOS/c≈Δλ/λ₀ for \|v\|≪c | Calibrated centroid shift Δλ and rest wavelength λ₀ in the same wavelength unit | m/s | Non-relativistic Doppler approximation; correct wavelength calibration and viewing geometry are required. |
| 12 | `tokamak_analysis.formula_library.ion_temperature_from_doppler_sigma` | T_i[eV]=(m_i c²/e)(σ_λ/λ₀)² | Instrument-corrected Gaussian σ_λ and λ₀ in the same wavelength unit; ion mass [kg] | eV | Thermal Gaussian Doppler width only; instrumental and non-thermal broadening must already be corrected. |
| 13 | `tokamak_analysis.formula_library.gaussian_fwhm_from_sigma` | FWHM=2 sqrt(2 ln 2) σ≈2.35482 σ | Gaussian standard deviation σ | same width unit | Gaussian profile only; sigma and FWHM are in the same unit. |
| 14 | `tokamak_analysis.formula_library.magnetic_energy_density` | u_B=B²/(2 μ₀) | B [T] | J/m³ = Pa | Vacuum magnetic energy-density approximation. |
| 15 | `tokamak_analysis.formula_library.toroidal_beta` | β_t=<p>/(B_t²/(2μ₀)) | Pressure p [Pa] and toroidal field B_t [T] | dimensionless | Local pressure or explicitly defined average; state averaging volume and field convention. |
| 16 | `tokamak_analysis.formula_library.safety_factor_large_aspect_ratio` | q(r)≈r B_t/(R B_p) | r,R [m], B_t,B_p [T] | dimensionless | Approximate circular, large-aspect-ratio geometry; equilibrium reconstruction is preferred. |
| 17 | `tokamak_analysis.formula_library.faraday_induction` | V_coil = −N dΦ_B/dt | Magnetic-flux derivative dΦ_B/dt [Wb/s] and positive integer coil turns N | V | Coil polarity/orientation and electronics response must be documented; Φ_B is the flux through one turn. |
| 18 | `tokamak_analysis.formula_library.plasma_loop_resistance_proxy` | R_p=V_loop/I_p | V_loop [V], I_p [A] | Ω | Signed ratio; consistent sign/time conventions are required and inductive terms can complicate interpretation. |
| 19 | `tokamak_analysis.formula_library.ohmic_power_approximation` | P_Ω≈V_loop I_p | V_loop [V], I_p [A] | W | Signed product; not generally the same as net loss power or total input power. |
| 20 | `tokamak_analysis.formula_library.energy_confinement_time` | τ_E=W_th/P_loss | W_th [J], P_loss [W] | s | Use consistent boundary, time window and loss-power definition; P_loss must be positive. |
| 21 | `tokamak_analysis.formula_library.greenwald_density_reference` | n_G[10^20 m^-3]=|I_p|[MA]/(π a²[m²]) | Signed I_p [MA] (magnitude used), minor radius a [m] | 10^20 m^-3 and SI equivalent m^-3 | Empirical reference scaling, not a universal physical limit; uncertainty at I_p=0 is non-differentiable when current uncertainty is non-zero. |
| 22 | `tokamak_analysis.formula_library.debye_parameter` | N_D=(4π/3)n_e λ_D³ | n_e [m^-3], λ_D [m] | dimensionless particle count | Number of particles in a Debye sphere. |
| 23 | `tokamak_analysis.formula_library.electron_plasma_angular_frequency` | ω_pe=sqrt(n_e e²/(ε₀m_e)) | n_e [m^-3] | rad/s | Local electron density; cold-plasma definition. |
| 24 | `tokamak_analysis.formula_library.ion_saturation_particle_flux_proxy` | Γ_i≈\|I_sat\|/(e A_eff) | \|I_sat\| [A], effective collection area A_eff [m²] | m^-2 s^-1 | Current magnitude is used; conversion is a proxy valid only with a justified collection geometry/model. |
| 25 | `tokamak_analysis.formula_library.radiated_power_fraction` | f_rad=P_rad/P_in | P_rad and defined input power P_in [W] | dimensionless | Input-power convention and time averaging must be specified; ratios above unity are flagged for review. |
| 26 | `tokamak_analysis.formula_library.integrated_surface_power` | P_target=∫_A q_perp dA | Heat flux q_perp [W/m²] and matching surface-area weights [m²] | W | Surface mapping, geometry, heat-flux calibration and quadrature weights are required. |
| 27 | `tokamak_analysis.formula_library.apparent_feature_speed` | v_app=Δx/Δt | Calibrated projected displacement Δx [m] and frame interval Δt [s] | m/s | Signed projected apparent velocity, not necessarily plasma-flow velocity. |
| 28 | `tokamak_analysis.formula_library.gas_pumping_speed` | S=Q/p | Throughput Q [Pa m³/s] and pressure p [Pa] | m³/s | Gauge location, conductance and throughput convention matter; SI Q/p units are required. |
| 29 | `tokamak_analysis.formula_library.injected_particle_number` | N_injected=∫ Ṅ(t) dt | Particle injection rate Ṅ(t) [s^-1] and matching time coordinate t [s] | particles | Rate calibration and strictly increasing time base are required. |
| 30 | `tokamak_analysis.formula_library.particle_confinement_time_proxy` | τ_p=N_p/Ṅ_loss | Stored particle number N_p and loss rate Ṅ_loss [particles/s] | s | Requires consistent control volume and particle balance. |
| 31 | `tokamak_analysis.formula_library.relative_fluctuation_level` | δx_rel=RMS(x−mean(x))/\|mean(x)\| | Finite time series x with a nonzero, numerically stable mean | dimensionless | Population RMS after mean removal; window and detrending must be specified; unstable near zero mean. |
| 32 | `tokamak_analysis.formula_library.rms_signal` | x_RMS=sqrt(mean(x²)) | Finite signal samples | same as signal | DC component is included; subtract the mean first if AC-only RMS is intended. |
| 33 | `tokamak_analysis.formula_library.nyquist_frequency` | f_N=f_s/2 | Uniform sampling frequency f_s [Hz] | Hz | Frequencies above Nyquist alias unless suitable anti-alias filtering is applied before sampling/downsampling. |
| 34 | `tokamak_analysis.formula_library.cross_correlation_lag` | τ_peak=argmax_τ R_xy(τ) | Synchronized, uniformly sampled x(t), y(t) and sample interval Δt [s] | s | Returns the lag of maximum positive cross-correlation; positive lag means y lags x; correlation does not establish causality. |
| 35 | `tokamak_analysis.formula_library.magnitude_squared_coherence` | C_xy(f)=\|S_xy(f)\|²/(S_xx(f)S_yy(f)) | Synchronized, uniformly sampled channel time series and sampling frequency [Hz] | frequency bins [Hz] and coherence [0,1] | Welch estimator; estimator uncertainty depends on segment count, overlap, window and stationarity. |
| 36 | `tokamak_analysis.formula_library.bolometry_regularized_inverse` | x=(AᵀA+λ²I)⁻¹Aᵀy (Tikhonov estimate) | Line-integrated measurements y, geometry/response matrix A, positive regularization λ | Spatial emissivity estimate x | Ill-posed inverse problem; this implementation is unconstrained, retains/flags negative emissivity and propagates measurement uncertainty only when supplied. |
| 37 | `tokamak_analysis.formula_library.particle_balance_rate` | dN/dt=Ṅ_in−Ṅ_out+S−L | Input/output and source/loss rates [particles/s] | particles/s | dN/dt=Ṅ_in−Ṅ_out+S−L; define the control volume and avoid double-counting source/loss terms. |

## Shared helper contracts

| Helper | Contract | Main failure modes guarded |
|---|---|---|
| `_finite` | Convert a scalar to float and reject NaN/±infinity. | Non-finite scalar. |
| `_nonnegative` | Finite scalar constrained to x ≥ 0. | Negative or non-finite input. |
| `_positive` | Finite scalar constrained to x > 0. | Zero, negative, or non-finite input. |
| `_integrate_with_uncertainty` | Trapezoidal integral over strictly increasing coordinates; independent pointwise uncertainty propagation using trapezoid weights. | Shape mismatch, fewer than two samples, non-finite values, non-increasing coordinates, negative integrand or invalid uncertainty array. |

## Specific scientific and numerical caveats

1. **Electron plasma frequency and Debye length:** local homogeneous-plasma expressions. They do not infer density or temperature from an instrument and do not account for uncertainty in constants or model inadequacy.
2. **Cyclotron frequency:** non-relativistic single-particle magnitude; sign of B is intentionally discarded. Relativistic shifts, harmonics, and resonance geometry are outside this callable.
3. **Ion sound speed:** closure-dependent expression (c_s=\sqrt{e(T_e+\gamma_iT_i)/m_i}\). The choice of `gamma_i`, ion species/mass, and validity of the fluid closure are user responsibilities.
4. **Thermal pressure and stored energy:** ideal-gas pressure uses density and temperature in stated units. The `3/2 ∫p dV` stored-energy expression assumes an isotropic scalar thermal pressure and an appropriate volume coordinate/profile; it is not a general anisotropic-energy or equilibrium reconstruction.
5. **Doppler velocity/temperature:** wavelength inputs must share units. The temperature helper expects an already instrument-corrected thermal Gaussian standard deviation; it does not remove instrumental, Stark, Zeeman, opacity, or unresolved-flow broadening.
6. **Magnetic energy density and beta:** `B²/(2 μ₀)` is vacuum magnetic energy density. Beta is a local or explicitly averaged pressure-to-magnetic-pressure ratio, not a profile-integrated global beta unless the supplied pressure and field represent that definition. First-order uncertainty propagation for magnetic energy is singular at B=0 with nonzero field uncertainty and now raises instead of reporting a misleading zero.
7. **Safety factor:** `q ≈ r B_t/(R B_p)` is a large-aspect-ratio/circular-geometry estimate, not a Grad–Shafranov/EFIT reconstruction. Near-zero poloidal field makes the estimate ill-conditioned and is rejected only at exactly zero; users must assess sensitivity near zero.
8. **Faraday induction, loop resistance, and ohmic-power proxy:** signs follow the supplied signal convention. `V_loop/I_p` is a proxy and `V_loop I_p` is not a general net-loss-power estimate when inductive terms or sign conventions matter.
9. **Energy/particle confinement and Greenwald density:** the implementation uses |Ip| and a positive minor radius; this is an empirical reference, not a universal validated prediction. Current uncertainty at exactly zero current is non-differentiable under the absolute-value convention and is rejected when non-zero.
10. **Ion-saturation current to particle flux:** a model-dependent proxy requiring effective collection area and a valid sheath/collection model. The absolute-current derivative is non-differentiable at zero; a nonzero uncertainty at exactly zero current now raises rather than reporting a misleading first-order uncertainty.
11. **Radiated power fraction:** values above one are preserved with a quality flag; they require checking power definitions, calibration, and time windows rather than automatic clipping.
12. **Surface-power integration:** assumes supplied area weights correspond to the flux samples. Uncertainty propagation assumes independent flux and area errors and omits covariance/systematic effects.
13. **Apparent feature speed:** projected feature displacement divided by frame interval; it is not automatically plasma-flow velocity.
14. **Pumping speed and particle balance:** require consistent SI units and definitions of throughput, input, output, source, and loss. The balance output may legitimately be negative.
15. **Signal RMS / relative fluctuation:** RMS includes DC; relative fluctuation uses population RMS divided by absolute mean and rejects a mean too close to zero.
16. **Nyquist frequency:** only (f_s/2) for uniformly sampled data; it does not verify uniform sampling or anti-alias filtering of the source data.
17. **Cross-correlation lag:** positive lag is documented as y lagging x; the peak is not evidence of causality and is not accompanied by a confidence interval.
18. **Magnitude-squared coherence:** estimator is bounded in [0,1] and requires non-constant finite signals. The output contains no confidence interval and depends on stationarity, windowing, segment count, and preprocessing.
19. **Bolometry inverse:** unconstrained linear Tikhonov solution. Negative emissivity values are retained and counted; geometry/calibration, non-negativity, regularization bias, and model discrepancy require independent assessment.
20. **Uncertainty policy across the library:** independent-input propagation does not include covariance unless explicitly implemented. Where a result depends strongly on calibration, geometry, response functions, correlated samples, or systematic uncertainty, the returned uncertainty is conditional and must not be presented as total uncertainty.

## Regression hardening included with this audit

- Reject boolean turn counts in Faraday induction; booleans are not physically meaningful coil-turn inputs.
- Reject singular first-order uncertainty propagation for magnetic energy at zero field when field uncertainty is nonzero.
- Reject singular first-order uncertainty propagation for absolute ion-saturation current at zero current when current uncertainty is nonzero.
- Reject coherence estimates for constant signals and verify returned estimates are finite; clip only floating-point roundoff to the mathematical [0,1] bounds.
- Add regression tests for these edge cases, independent analytic checks for the mapped formula primitives, and an explicit model-limited test plus diagonal-matrix analytic reference for the bolometry inverse.

## What remains required before claiming full scientific verification

- Run the full CI matrix (Python 3.10, 3.11, 3.12) on the exact audited commit, including package installation, Ruff, pytest/coverage, Streamlit AppTest, and HTTP health check.
- The current main branch now has regression coverage touching all 37 mapped formula callables, including independent numerical references for the analytic primitives and analytic/property checks for coherence and bolometry. Continue expanding dimensional/unit tests and the remaining uncertainty branches, including covariance and systematic-error policies.
- Execute the complete UI/CLI workflow matrix with representative valid and invalid files, and assert no stale results survive failed imports or failed analysis.
- Validate each promoted diagnostic with a compatible independent experimental reference, archived fixture, provenance manifest, and quantitative discrepancy/uncertainty report.
- Keep diagnostic registry, formula registry, packaged registry copy, CLI/UI labels, and implementation-status metadata synchronized.
