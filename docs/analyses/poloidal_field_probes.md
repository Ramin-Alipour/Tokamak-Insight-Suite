# Poloidal-field coils / magnetic probes

## Supported workflow

The Probe Analysis workspace accepts one or multiple digitized poloidal-field probe channels and a time column. For each channel, provide the documented acquisition-chain calibration

\[
B_p(t) = g\,s(t) + b
\]

where \(g\) is the gain in tesla per digitizer unit and \(b\) is the offset in tesla. The optional baseline window subtracts the mean calibrated field over the selected interval. Outputs include calibrated field traces, \(dB_p/dt\) from nonuniform-grid finite differences, channel statistics, median sample interval/rate, relative sample-interval deviation, and optional propagated pointwise uncertainty for the field and derivative. JSON export preserves the calculated outputs and stated limitations.

## Inputs and validation

- Time in seconds: finite, one-dimensional, at least three samples, strictly increasing.
- Signal: finite array with shape (samples,) or (samples, channels).
- Gain: finite, nonzero scalar or one value per channel; offset: finite scalar or one value per channel.
- Optional signal uncertainty: non-negative, finite 1-sigma uncertainty in digitizer units, broadcastable to the signal shape.
- Optional baseline interval: finite increasing start/stop values containing at least two samples.
- Optional channel names: one non-empty string per channel.

Sampling variation above 1% emits a warning. A sampling-rate estimate alone does not establish anti-aliasing or adequate bandwidth.

## Physical scope and limitations

This is calibrated local-signal analysis, not a PF-coil current inversion or equilibrium reconstruction. It does not infer poloidal flux, plasma boundary, coil geometry, transfer function, absolute calibration, or spatial gradients. Derivative uncertainty uses the finite-difference operator and independent raw-sample errors. Baseline-corrected field uncertainties include the shared estimated baseline mean in marginal variances; covariance between corrected samples is not returned. Calibration coefficient uncertainty and timebase uncertainty are not currently propagated. A device-specific calibration and independent experimental validation remain necessary.

## Acceptance

Automated acceptance tests cover multi-channel calibration, derivative units and values, uncertainty shape, invalid input rejection, and the interactive Streamlit path including JSON export. These tests establish software behavior only, not device-specific scientific validation.
