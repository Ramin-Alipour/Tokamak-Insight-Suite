# Microwave Collective Thomson Scattering (CTS)

## Scope

The CTS module is a frequency-domain microwave CTS analysis path. It is separate from optical Thomson-scattering Analysis 1 and must not be described as an optical spectral fit or a complete electromagnetic CTS simulator.

The implemented forward model is an unmagnetized, electrostatic, longitudinal multi-species dielectric response. It computes the electron and ion susceptibilities, the longitudinal dielectric function, the dynamic spectral density and convolution with a measured frequency-response matrix.

It does not include a full magnetized electromagnetic scattering tensor, ray tracing, beam overlap, polarization coupling, receiver transfer function or absolute received-power calibration.

## Inputs

- Calibrated frequency-shift axis in Hz.
- Calibrated non-negative CTS spectrum.
- Probe frequency in Hz.
- Scattering geometry, either a scalar scattering angle or incident/scattered unit vectors.
- Explicit ion species metadata (name, charge state and mass in amu).
- Explicit ion charge-density fractions summing to unity.
- Explicit electron-flow projection along the scattering wavevector. No stationary-electron-flow default is applied; enter 0 only when that assumption is intentionally selected.
- Measured frequency-response matrix.
- Optional per-bin 1-sigma uncertainty or a full positive-definite frequency-bin covariance matrix.
- Optional response-matrix uncertainty.
- Optional uncertainties for probe frequency, scalar scattering angle, electron-flow projection and ion fractions.
- Optional fast-ion model.

The current background-ion model uses one common ion temperature and one common projected ion flow for all supplied ion species. Species-specific ion temperatures and species-specific flow inference are not implemented. Ion species, charge states, masses and fractions must be supplied explicitly; the interface does not prefill D+ or a unit fraction. The result returns the exact externally supplied electron-flow projection and its optional input uncertainty, explicitly marked as not fitted; inferred ion-flow uncertainty remains a separate fitted-parameter result.


## Magnetized full-electromagnetic forward model

A separate forward model and inversion are available as `tokamak_analysis.cts_full_em.full_em_cts_spectral_density` and `tokamak_analysis.cts_full_em.analyze_full_em_cts_spectrum` (also re-exported by `tokamak_analysis.cts_analysis`). `analyze_cts_spectrum_model(model="electrostatic"|"full_electromagnetic", ...)` dispatches explicitly; it does not replace or silently alter the existing longitudinal electrostatic model.

The implemented local kinetic response solves the linearized, collisionless Vlasov equation for isotropic non-relativistic Maxwellian species in a uniform magnetic field. Gyro-harmonics are retained explicitly and the velocity integrals use Gauss-Laguerre (perpendicular) and Gauss-Hermite (parallel) quadrature. The species conductivity tensors are assembled into the dielectric tensor and Maxwell wave tensor. Independent-particle current fluctuations are generated species-by-species with the classical fluctuation-dissipation relation, screened through the inverse wave tensor, and propagated into a joint covariance of electron density, electron current, electric field and magnetic field. The returned covariance retains their cross-correlations rather than summing independent fluctuation powers. The inversion profiles a multiplicative spectral amplitude and constant background as nuisance parameters and reports a local Jacobian covariance.

Required inputs are the vector magnetic field, incident/scattered beam directions, complex transverse incident/scattered polarization vectors, electron flow vector, and a flow vector for every ion species. The inversion additionally requires measured spectral uncertainty or a full spectral covariance and a measured instrument-response matrix; it fits electron density, electron temperature and one common ion temperature, while all flow vectors and composition remain fixed external inputs. No vector direction, polarization, or ion flow is silently assumed. The harmonic order, velocity quadrature order and positive numerical damping rate are explicit numerical controls; the default damping is a numerical Landau-contour regularization, not a measured collision frequency.

**Scope boundary:** this is a local magnetized electromagnetic kinetic fluctuation solver, not yet a reproduction of the complete Bindslev three-wave CTS transfer formulation. In particular, the receiver coupling currently projects the screened electron-current radiation with the supplied polarization factor; it does not yet include the complete density/current/E/B three-wave coupling tensors and their incident/scattered cold-plasma mode normalization. It also does not include ray tracing, finite beam overlap, relativistic or anisotropic fast-ion VDFs, or spatially varying magnetic geometry. These omissions are returned in the result metadata and must not be described as completed Full-EM diagnostic physics.

Synthetic regression tests check finite non-negative spectra, covariance Hermiticity/positive semidefiniteness, and mandatory metadata errors. They are software consistency tests, not experimental validation.


## Inversion strategies

### Differential evolution

The default path uses differential evolution for global initialization followed by bounded least-squares refinement.

### Improved real-coded genetic search

The improved_genetic option uses a deterministic seeded real-coded genetic search with elitism, tournament selection, blend crossover and an adaptive mutation scale, followed by bounded least-squares refinement.

This is an independent adaptive genetic implementation informed by Zhang et al., Review of Scientific Instruments 95, 093542 (2024), DOI 10.1063/5.0215734. It is not claimed to reproduce the exact paper-specific fitness function, parameter sensitivity calibration or genetic operators without the full article's implementation details.

### Regularized inversion

The regularized option is restricted to inferred tabulated fast-ion VDF reconstruction. It adds a second-difference smoothness penalty to the normalized non-negative VDF only. It does **not** impose a prior or penalty around the initial electron density, electron temperature, ion temperature, ion flow, or fast-ion fraction; those initial values are optimizer starting points. The regularization strength and VDF smoothness scale are explicit user-supplied controls and should be varied in synthetic recovery and sensitivity tests; no universal default lambda is scientifically justified.

The 2026 regularized Thomson-inversion literature motivates regularization to suppress data-consistent but physically implausible oscillatory solutions. The TIS regularized CTS objective is a scoped implementation, not an exact reproduction of that optical Thomson time-series method.

## Fast-ion representations

- Hot Maxwellian: fits a fast-ion fraction and temperature using a two-population Maxwellian approximation.
- Fixed tabulated 1-D VDF: accepts a supplied non-negative velocity distribution and fits its fraction; the VDF shape is external.
- Inferred tabulated 1-D VDF: fits a non-negative normalized tabulated VDF using a softmax mass parameterization and second-difference regularization. The VDF-node covariance is obtained by transforming the local logit covariance.

The inferred VDF is a one-dimensional projection along the scattering wavevector. Its reconstruction depends on the selected velocity grid, initial seed distribution, noise covariance and regularization strength. It is not a unique unrestricted three-dimensional velocity-distribution measurement.

The tabulated susceptibility uses a Landau-contour evaluation of the longitudinal kinetic susceptibility, with PCHIP interpolation and principal-value quadrature for the supplied one-dimensional VDF.

## Uncertainty and model discrepancy

When supplied, the full frequency-bin covariance is used to whiten residuals in the objective. Response-matrix uncertainty and supported external parameter uncertainties are propagated by Monte Carlo refits. The Monte Carlo success count and requested count are returned; the uncertainty contribution is not added when too few perturbation fits succeed. Ion-fraction uncertainties are currently perturbed independently by species and renormalized to sum to one; a full species-composition covariance matrix is not accepted, so correlations between fraction uncertainties cannot be represented.

Excess reduced chi-square above unity is exposed as a separate model-discrepancy covariance component. This is conditional on the supplied noise covariance and chosen forward model; it is not independent experimental validation.

The output includes fit covariance, parameter provenance, Jacobian rank and condition number, response/external Monte Carlo success counts, model-discrepancy diagnostics and explicit model limitations.

## Synthetic software tests

The CTS test suite covers:

- Positive finite forward spectra and geometry sensitivity.
- Recovery of known electron density, electron temperature, ion temperature and ion-flow projection from synthetic spectra.
- Hot-Maxwellian and tabulated non-Maxwellian fast-ion forward models.
- Deterministic and bounded genetic-search behavior.
- Full frequency-bin covariance whitening.
- Regularized inversion and regularized tabulated VDF reconstruction.
- Response-matrix and supported external-parameter Monte Carlo propagation.

These are mathematical and software regression tests using controlled synthetic spectra. They are not empirical validation of a physical CTS diagnostic.

## Scientific references

1. J. Sheffield, D. Froula, S. H. Glenzer and N. C. Luhmann Jr., *Plasma Scattering of Electromagnetic Radiation*, 2nd ed., Academic Press, 2011.
2. J. Zhang et al., “Spectral decomposition for collective Thomson scattering based on an improved genetic algorithm,” *Review of Scientific Instruments* 95, 093542 (2024), DOI: 10.1063/5.0215734.
3. B. C. Foo et al., “Regularized inversion of optical Thomson scattering spectra for improved determination of plasma parameter evolution,” *Review of Scientific Instruments* 97, 103535 (2026), DOI: 10.1063/5.0350497.
4. G. A. Baker et al., “Recovering Non-Maxwellian Velocity Distribution Functions from Collective Thomson Scattering Spectra Using Generalized Basis Representations,” arXiv:2610.37516 (2026). This preprint motivates flexible non-Maxwellian VDF inversion; TIS currently uses a tabulated 1-D projected VDF with explicit smoothness regularization.


## Magnetized electromagnetic CTS references

7. X. Fang, D. Xia, M. Huang, F. Zhang and Y. Ding, “Study on the Full Electromagnetic Model of Collective Thomson Scattering in Magnetic Confinement Fusion,” *Power Generation Technology* **46**(1), 200–208 (2025), DOI: [10.12096/j.2096-4528.pgt.24149](https://doi.org/10.12096/j.2096-4528.pgt.24149). This is the primary recent reference for the magnetized full-electromagnetic CTS formulation, including density, current, electric-field and magnetic-field fluctuations and their cross-correlations.

8. H. Bindslev, “A quantitative study of scattering from electromagnetic fluctuations in plasmas,” *Journal of Atmospheric and Terrestrial Physics* **58**(8–9), 983–989 (1996), DOI: [10.1016/0021-9169(95)00129-8](https://doi.org/10.1016/0021-9169(95)00129-8). Foundational transfer/coupling formulation for correlated electromagnetic plasma fluctuations.

## Recent diagnostic and inversion literature

5. T. Verdier et al., “Robust monitoring of thermal and fast ions using collective Thomson scattering: Combining physics- and data-driven background estimation,” *Review of Scientific Instruments* **96** (2025), 043513. DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948). This supports the importance of CTS background characterization; TIS does not perform automatic background estimation because the project input contract assumes prepared, calibrated measurement data.
6. “Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas,” *Fusion Engineering and Design* (2026). DOI: [10.1016/j.fusengdes.2026.115845](https://doi.org/10.1016/j.fusengdes.2026.115845). This recent work combines ray tracing, forward modelling and Bayesian spectral inversion for CTS diagnostic design. It is cited as current diagnostic/inversion context, not as a claim that TIS reproduces its full magnetized electromagnetic model.


The regularized inversion option is intentionally restricted to the inferred tabulated fast-ion velocity-distribution workflow. It applies smoothness regularization to that inferred distribution only; user-supplied initial plasma values remain optimizer starting points, not implicit Bayesian priors. Response-matrix and external-parameter Monte Carlo refits use the same regularized residual as the central fit.


## Cold-plasma propagation and finite-beam geometry

`tokamak_analysis.cts_em_geometry` adds the high-frequency wave-propagation layer used by the CTS incident and receiver beams:

- Stix (S,D,P) cold-plasma dielectric tensor in arbitrary laboratory-frame magnetic-field orientation, using signed species cyclotron frequencies and the `exp(-i omega t)` convention.
- Cold-plasma Maxwell wave operator and its three Hermitian eigenbranches/electric eigenvectors.
- Hamiltonian geometrical-optics ray tracing through an explicitly supplied 3-D equilibrium callback. The callback must return the local magnetic-field vector and every species density; no equilibrium is fabricated.
- Analytic overlap integral for two normalized three-dimensional Gaussian beam-intensity distributions, including finite beam widths and non-coincident beam centres.

These high-frequency propagation/geometry components are now accompanied by the Bindslev low-temperature three-wave source-current tensor and the full correlated (n,v,E,B) covariance contraction described below. The ray tracer is geometrical optics and does not model diffraction, mode conversion at branch degeneracies, or resonant absorption layers. It must not be interpreted as a full-wave solver. The relativistic fast-ion linear response is supported, but the nonlinear three-wave vertex itself remains the low-temperature Bindslev closure.

The cold-plasma tensor follows the Stix formulation documented by PlasmaPy and the high-frequency cold-wave / low-frequency hot-kinetic separation used by Fang et al. (2025). Software tests cover the vacuum limit, Hermiticity, Gaussian overlap normalization and straight-ray propagation in vacuum; they are numerical consistency tests, not experimental validation.



## Bindslev three-wave coupling and relativistic fast-ion extension

The full-EM forward path now applies the magnetized low-temperature Bindslev source-current closure rather than projecting only the electron-current covariance onto the receiver polarization.

Implemented in `tokamak_analysis/cts_bindslev.py`:

- Rank-four `Gamma_{abij}` tensor and its explicit inverse, with the inverse identity checked numerically.
- Rank-three `X_{ijk}` pseudotensor generated from the inverse-`Gamma` contraction in Bindslev's magnetized source-current formulation.
- Incident-wave density, particle-flux and magnetic perturbations derived from the cold-plasma susceptibility and Maxwell-Faraday relations.
- Full source-current map from the ten fluctuation components `[delta_n, delta_v(3), delta_E(3), delta_B(3)]`.
- Receiver projection and contraction with the complete Hermitian covariance, retaining every auto-spectrum and complex cross-spectrum. The density, particle-flux, electric-field and magnetic-field contributions are separately returned for diagnostics.

This implements the low-temperature magnetized three-wave closure represented by Bindslev JET-P(93)41 Eqs. (98)-(102), (111), and (114)-(136), consistent with the correlated electromagnetic fluctuation transfer principle of Bindslev (1996). The cold high-frequency susceptibilities are evaluated separately at incident and scattered frequencies. This is not a claim that the entire relativistic nonlinear three-wave vertex has been derived: the Bindslev low-temperature source-current closure remains the coupling approximation.

Implemented in `tokamak_analysis/cts_relativistic.py`:

- Relativistic gyrotropic momentum-space Vlasov response with exact (v=p/(\gamma m)) and Ω/γ gyrofrequency.
- Anisotropic relativistic bi-Gaussian momentum distributions with independently specified perpendicular and parallel temperatures.
- User-supplied tabulated gyrotropic distributions (f(p_\perp,p_\parallel)), normalized over momentum space.
- Fourier-collocation solution in gyrophase, including the full relativistic Lorentz force.
- Non-equilibrium fast-ion fluctuation source from the relativistic Klimontovich orbit shot-noise spectrum, rather than applying an equilibrium fluctuation-dissipation theorem to an anisotropic population.
- Fast-ion populations enter the total dielectric response and the screened fluctuation covariance through the explicit `fast_ion_populations` input.

Each fast-ion population requires its name, charge state, mass, density, flow vector and either a `relativistic_bigaussian` distribution (perpendicular/parallel temperatures) or a `tabulated_gyrotropic` distribution with momentum grids and a non-negative distribution array. No fast-ion density, anisotropy or shape is inferred by default. The inversion currently fits (n_e,T_e,T_i); fast-ion distribution parameters are supplied externally, not fitted.

Important scientific boundary: the relativistic fast-ion *linear response and particle-noise spectrum* are implemented, but the nonlinear Bindslev source-current vertex itself remains the low-temperature closure. A relativistic second-order kinetic vertex, including relativistic momentum derivatives and all corresponding fluctuation cross-correlations, is not claimed as complete. Bulk thermal species also remain on the existing non-relativistic Maxwellian kinetic path. Tests are numerical consistency/regression tests, not experimental validation.

References:
- H. Bindslev, “Three Wave Mixing and Thomson Scattering in Plasmas,” JET-P(93)41 (1993), detailed kinetic source-current derivation, including the `Gamma` and `X` tensors and the microscopic-fluctuation transfer equation.
- H. Bindslev, “A quantitative study of scattering from electromagnetic fluctuations in plasmas,” *Journal of Atmospheric and Terrestrial Physics* **58** (1996), 983–989, DOI: [10.1016/0021-9169(95)00129-8](https://doi.org/10.1016/0021-9169(95)00129-8).
- H. Bindslev, “Dielectric effects on Thomson scattering in a relativistic magnetized plasma,” *Plasma Physics and Controlled Fusion* **33** (1991), 1775–1804, DOI: [10.1088/0741-3335/33/14/005](https://doi.org/10.1088/0741-3335/33/14/005).
