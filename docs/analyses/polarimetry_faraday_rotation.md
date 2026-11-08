# Polarimetry / Faraday rotation (analysis 14)

## Implemented scope

The analysis accepts **calibrated** vacuum wavelength and signed Faraday rotation in SI units and evaluates

\[
\theta = C_F \lambda^2 \int n_e B_{\parallel}\,dl,
\qquad
C_F = \frac{e^3}{8\pi^2\epsilon_0 m_e^2 c^3}.
\]

It returns the signed line integral of electron density times the parallel magnetic field. If both electron density and chord length are supplied, it also returns a chord-averaged parallel field, assuming the supplied density is uniform along the path. Optional independent uncertainties in rotation, wavelength, density and path length are propagated to first order.

## Dashboard

Open **Advanced Analysis → Faraday-rotation polarimetry**. Enter comma-separated wavelength values (m) and signed rotation values (rad). Optional scalar density (m⁻³) and path length (m) enable the average-field estimate. The workflow exports CSV results and a JSON result/limitations record.

## Input and interpretation requirements

- Wavelength must be calibrated and positive; rotation must be calibrated, finite and signed consistently with the instrument handedness and viewing direction.
- Rotation and wavelength arrays must be non-empty, one-dimensional and the same length.
- Density and path length must be supplied together and be finite and positive.
- Uncertainties are standard (1-sigma), finite and non-negative. The propagated result assumes independent errors.
- A single chord produces a line integral, **not** a spatially resolved magnetic field or equilibrium reconstruction.
- This module does not calibrate raw polarimeter electronics, model refraction or additional birefringence, infer density, or perform multi-chord/ray-tracing inversion. Device-specific calibration and coordinate/sign conventions remain user responsibilities.

The analysis is complete within this declared computational scope; it is not a claim of independent experimental validation.
