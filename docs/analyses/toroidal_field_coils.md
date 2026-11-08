# Toroidal-field coils / magnetic probes (analysis 8/80)

## Scope

Two explicit computational workflows are supported: calibrated field/current/directly calibrated signal analysis, and integration of calibrated pickup-coil voltage when a validated sensitivity and initial-field boundary condition are supplied. This is not a coil-design solver or equilibrium reconstruction.

## Workflow A — calibrated signal

For a documented input signal s(t):

`B_t(t) = k_cal [s(t) - s_0]`

For field input, the signal is already in tesla and the calibration coefficient must equal 1. For current input, the coefficient is T/A. For directly calibrated voltage, it is T/V. Raw pickup-coil voltage must not be passed as if it were instantaneous field.

The workflow reports the calibrated trace, mean, RMS (including any remaining DC component), sample standard deviation, peak-to-peak range, sampling rate based on the median sample interval, sampling interval jitter, and optional first-order independent signal/calibration uncertainty.

## Workflow B — pickup-coil voltage integration

Function: `tokamak_analysis.toroidal_field_coils.integrate_pickup_coil_voltage`.

When a validated transfer coefficient S in V/(T/s) is valid over the measured bandwidth:

`V_corrected(t) = S dB/dt`

The code calculates the field change by cumulative trapezoidal integration, then applies the required boundary condition:

`B(t) = B(t_0) + integral[V_corrected(t)/S] dt`

Required inputs are finite, aligned voltage and strictly increasing time arrays, non-zero validated sensitivity, and `initial_field_T`. An optional finite voltage baseline and non-negative pointwise voltage uncertainty, sensitivity uncertainty, and initial-field uncertainty are accepted. The uncertainty estimate propagates independent pointwise voltage noise through the trapezoidal increments and combines it with sensitivity and initial-field uncertainty in quadrature.

## Validation and limitations

Focused tests cover calibrated conversion, uncertainty propagation, malformed inputs, prohibition of implicit voltage integration, field-mode calibration, pickup-voltage integration, boundary conditions, and invalid uncertainty inputs.

No coil geometry, mutual inductance, plasma-current contamination, saturation, ferromagnetic response, or frequency-dependent transfer function is inferred automatically. Integration drift from voltage offsets is not automatically removed; baseline uncertainty and correlated drift are excluded from the uncertainty estimate. Sensitivity and initial field must be independently established for each instrument and shot. Sampling-rate/jitter metrics are not a complete bandwidth, aliasing or channel-synchronization audit. Local magnetic measurements do not by themselves provide Grad–Shafranov equilibrium or machine-wide field reconstruction.

“Complete” refers only to the declared computational scope. Device-specific calibration and independent experimental validation remain necessary.
