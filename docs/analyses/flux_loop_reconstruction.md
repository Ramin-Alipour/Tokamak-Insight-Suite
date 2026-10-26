# Flux-loop magnetic-flux reconstruction (analysis 11)

## Scope

The flux-loop workflow reconstructs magnetic-flux change from an already calibrated loop-voltage time series. It applies Faraday's law,

\[
V_{\mathrm{loop}}=-N\frac{d\Phi}{dt},\qquad
\Delta\Phi(t)=-\frac{1}{N}\int_{t_0}^{t}s\,[V_{\mathrm{meas}}(t')-V_{\mathrm{offset}}]dt',
\]

where \(N\) is the positive integer number of turns and \(s\in\{-1,+1\}\) is the wiring-polarity correction. Integration uses the trapezoidal rule on the supplied time coordinates, including nonuniform sampling. Reconstructed flux is \(\Phi(t)=\Phi_0+\Delta\Phi(t)\).

## Dashboard

Open **Advanced Analysis → Flux-loop magnetic-flux reconstruction**. Load a CSV/table containing time in seconds and calibrated loop voltage in volts, select distinct columns, and specify turns, polarity, constant offset, and the initial-flux integration constant. An optional sampling-gap guard rejects time intervals above the chosen maximum. The workflow displays corrected voltage and reconstructed flux, sample/voltage diagnostics, and exports the numerical results as CSV and JSON.

## Inputs and outputs

- Required: finite, one-dimensional, matching time and voltage arrays with at least two samples; time must be strictly increasing.
- Optional: positive integer turns (default 1), finite initial flux in Wb (default 0), finite voltage offset in V (default 0), polarity +1 or −1 (default +1), and positive maximum allowed sample gap in seconds.
- Outputs: corrected voltage, cumulative voltage integral, flux change, flux including the specified initial condition, sample-interval statistics, voltage statistics, and peak-to-peak flux change.

## Assumptions and limitations

- Input voltage has already been calibrated in volts and the time base is in seconds.
- A constant offset is adequate when offset correction is used. Baseline drift is not automatically estimated.
- Zero initial flux is only a reference integration constant, not a measurement of absolute tokamak flux.
- No voltage/time calibration uncertainty, covariance, geometry/orientation, bandwidth, or baseline-drift uncertainty is propagated.
- This workflow does not infer plasma current and is not a Grad–Shafranov/EFIT equilibrium solver.
- Software tests establish numerical behavior under the declared assumptions; they do not establish device-specific calibration or independent experimental validation.

## Acceptance tests

The core tests cover Faraday-law sign and turns, polarity/offset handling, nonuniform time steps, invalid and non-finite inputs, configuration validation, maximum-gap rejection, zero-voltage behavior, and JSON-friendly serialization. The Streamlit AppTest verifies that the workflow is selectable and renders its missing-data guidance without raising an exception. Both are included in the Python 3.10–3.12 GitHub Actions CI matrix.
