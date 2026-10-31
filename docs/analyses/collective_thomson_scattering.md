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

## Inversion strategies

### Differential evolution

The default path uses differential evolution for global initialization followed by bounded least-squares refinement.

### Improved real-coded genetic search

The improved_genetic option uses a deterministic seeded real-coded genetic search with elitism, tournament selection, blend crossover and an adaptive mutation scale, followed by bounded least-squares refinement.

This is an independent adaptive genetic implementation informed by Zhang et al., Review of Scientific Instruments 95, 093542 (2024), DOI 10.1063/5.0215734. It is not claimed to reproduce the exact paper-specific fitness function, parameter sensitivity calibration or genetic operators without the full article's implementation details.

### Regularized inversion

The regularized option adds a scaled prior penalty around the supplied initial parameter vector. For inferred tabulated fast-ion VDFs, it also adds a second-difference smoothness penalty on the normalized non-negative VDF. The regularization strength is user-supplied and should be varied in synthetic recovery and sensitivity tests; no universal default lambda is scientifically justified.

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


## Recent diagnostic and inversion literature

5. T. Verdier et al., “Robust monitoring of thermal and fast ions using collective Thomson scattering: Combining physics- and data-driven background estimation,” *Review of Scientific Instruments* **96** (2025), 043513. DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948). This supports the importance of CTS background characterization; TIS does not perform automatic background estimation because the project input contract assumes prepared, calibrated measurement data.
6. “Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas,” *Fusion Engineering and Design* (2026). DOI: [10.1016/j.fusengdes.2026.115845](https://doi.org/10.1016/j.fusengdes.2026.115845). This recent work combines ray tracing, forward modelling and Bayesian spectral inversion for CTS diagnostic design. It is cited as current diagnostic/inversion context, not as a claim that TIS reproduces its full magnetized electromagnetic model.
