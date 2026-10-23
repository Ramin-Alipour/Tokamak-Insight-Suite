# Toroidal-field coils / magnetic probes (analysis 8/80)

## Scope
Convert sampled, documented coil signals to toroidal-field estimates using an explicit calibration coefficient. This is not a coil-design solver or equilibrium reconstruction.

## Relation
B_t(t) = k_cal [s(t) - s_0]. For current input, k_cal is T/A; for directly calibrated voltage, T/V. A pickup-coil voltage generally relates to changing linked flux and must not be treated as instantaneous field without a validated transfer function or justified integration.

## Inputs and outputs
Inputs: strictly increasing time (s), finite signal, signal kind (field/current/voltage), explicit calibration coefficient, optional baseline and uncertainties. Outputs: B_t(t) (T), mean, RMS, standard deviation, peak-to-peak, sampling rate/jitter, and optional uncertainty.

## Uncertainty
For independent signal and calibration errors: sigma_B^2 = (k_cal sigma_s)^2 + ([s-s_0] sigma_k)^2. Baseline uncertainty, correlated errors, bandwidth response and calibration-model discrepancy are excluded.

## Limitations
No automatic conversion from coil current using assumed geometry; no inference of total toroidal field from current alone; no compensation for mutual inductance, plasma current, ferromagnetic structures, saturation or frequency response. Verify polarity and calibration against an independent reference.
