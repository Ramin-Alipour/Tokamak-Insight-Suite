# Thomson Scattering (Analysis 1 of 80)

## Scope and completion status

**Complete within the declared geometry-independent software scope.** Analysis 1 consumes already-calibrated electron-density and electron-temperature profiles together with their supplied calibrated spatial coordinate. It does not calibrate raw detector data, infer device dimensions, request optical collection geometry, or calculate geometry-weighted volume averages.

The analysis has two supported workflows:

1. **Calibrated profile analysis** — profile extrema and their supplied-coordinate locations, measured-interval line averages, radial gradients, normalized gradients, local scale lengths, and optional first-order uncertainty propagation.
2. **Within-domain profile reconstruction** — interpolation/smoothing of the supplied calibrated profiles on a denser coordinate grid, restricted to the measured coordinate interval.

Raw-spectrum fitting and density inversion from scattered optical power are not part of Analysis 1. They require additional spectral/instrument or optical-geometry inputs and are intentionally excluded from this analysis contract and its dashboard workflow list. Their underlying modules/tests, where present, do not change the scope of Analysis 1.

No independent experimental validation is claimed. Automated tests check software behavior for specified inputs; they do not establish experimental validity.

## Calibrated profile input contract

| Input | Unit | Requirement |
|---|---|---|
| `radius_m` | m | Required; supplied calibrated spatial coordinate, one-dimensional, finite, unique |
| `density_m3` | m⁻³ | Required; calibrated electron-density profile, finite and non-negative |
| `temperature_ev` | eV | Required; calibrated electron-temperature profile, finite and non-negative |
| `density_uncertainty_m3` | m⁻³ | Optional pointwise 1-sigma uncertainty, finite and non-negative |
| `temperature_uncertainty_ev` | eV | Optional pointwise 1-sigma uncertainty, finite and non-negative |

The three required arrays must have equal lengths and at least two samples. Samples are sorted by the supplied coordinate, and all associated profile and uncertainty arrays are reordered consistently. Interpolation targets must be a non-empty, one-dimensional array of finite real numeric values; Boolean and string values are rejected. The coordinate is a supplied data axis, not a machine dimension inferred by TIS. The measured interval need not cover the entire plasma.

## Calculations and equations

For calibrated density \(n_e(r)\) and electron temperature \(T_e(r)\), the numerical gradients are

\[
\frac{dn_e}{dr}, \qquad \frac{dT_e}{dr}.
\]

The normalized gradients are

\[
\frac{1}{n_e}\frac{dn_e}{dr}, \qquad
\frac{1}{T_e}\frac{dT_e}{dr},
\]

and the local scale lengths are

\[
L_{n_e}=-\frac{n_e}{dn_e/dr}, \qquad
L_{T_e}=-\frac{T_e}{dT_e/dr}.
\]

A normalized gradient is undefined where its corresponding profile value is zero. A local scale length is reported as undefined (NaN) where the profile value is zero or the numerical gradient is unresolved. Gradient resolvability is assessed pointwise against a floating-point round-off bound derived from the absolute terms in that location's finite-difference calculation; a large gradient elsewhere in the profile does not set the local threshold. Derivatives use numerical finite differences on the supplied coordinate.

The line average for a profile \(f(r)\) over the measured interval \([r_0,r_N]\) is calculated by trapezoidal integration:

\[
\langle f\rangle_{\mathrm{line}} =
\frac{1}{r_N-r_0}\int_{r_0}^{r_N}f(r)\,dr.
\]

This is an average over the measured coordinate interval, not a geometry-weighted volume average and not automatically a full-plasma average.

## Outputs

- Sorted supplied coordinate, density and temperature profiles.
- Minimum and maximum density and temperature values and their locations on the supplied coordinate.
- Coordinate range and measured-interval line averages.
- Density and temperature gradients.
- Normalized density and temperature gradients.
- Local density and temperature scale lengths.
- When corresponding pointwise uncertainties are provided: propagated uncertainties for line averages, gradients, normalized gradients and scale lengths.
- For the optional reconstruction workflow: reconstructed profiles and gradients only within the measured coordinate interval, with method and domain metadata. The output explicitly reports that reconstructed-profile uncertainty and fit confidence intervals are not estimated; input uncertainties used as spline weights are not output uncertainties.

## Uncertainty assumptions and limitations

- Input profiles and coordinates are assumed calibrated; TIS does not perform calibration in this analysis.
- Optional uncertainty propagation assumes independent pointwise measurement errors.
- For smoothing-spline reconstruction, supplied pointwise uncertainties are used only as inverse-sigma fit weights. They are not propagated to the reconstructed profile values; reconstructed-profile uncertainties and fit confidence intervals are explicitly unavailable. Weighting a fit must not be interpreted as an uncertainty estimate for its output.
- Covariance between samples, calibration uncertainty, model discrepancy, and peak-location uncertainty are not estimated.
- Numerical derivatives can amplify noise, especially for sparse or noisy profiles.
- No extrapolation beyond the measured coordinate interval is performed.
- No machine dimensions, major radius, optical collection geometry, equilibrium reconstruction, or geometry-weighted volume averages are inferred.
- No independent experimental validation is claimed.

## Completion gate

Analysis 1 is complete for the stated software scope: calibrated profile analysis and reconstruction within the measured coordinate interval. The dashboard and diagnostic registry should expose only these two workflows under Analysis 1. Raw-spectrum fitting and scattered-power density inversion are outside this contract and must not be presented as part of its geometry-independent workflow.
