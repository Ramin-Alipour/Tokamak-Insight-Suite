# Mirnov / pickup-coil analysis

## Supported calculations

The Mirnov workflow provides:
- validated uniformly sampled voltage input and sampling/Nyquist metadata;
- voltage statistics and Welch power spectral density (V²/Hz), with configurable segment length and detrending;
- dominant spectral peak candidates, optional frequency-band restriction, and peak-prominence threshold;
- ideal fixed-coil estimate of dB/dt when effective area and integer turn count are supplied;
- trapezoidal integration of calibrated coil voltage to a relative magnetic perturbation, with explicit baseline and polarity controls;
- multi-channel cross-spectral phase/coherence and an alias-prone toroidal mode-number candidate using supplied coil angles.

## Physical relations

For a stationary coil with known effective area and orientation, the ideal relation is (V=-N A,dB/dt). The inferred derivative is only as reliable as voltage calibration, effective area, turns, orientation and the coil/electronics transfer function. Integration yields a perturbation relative to the supplied initial value, not the absolute equilibrium field.

Welch PSD is reported in V²/Hz. Peak frequencies are candidate oscillations; they are not automatically classified as MHD modes. Multi-channel mode-number estimation uses cross-spectral phase at the reference channel's strongest non-DC spectral bin and searches integer candidates from (-N_{channels}) through (+N_{channels}). Spatial aliasing, channel polarity, geometry and mode structure can make this estimate ambiguous.

## Input requirements and limitations

- Time and signal arrays must be finite, one-dimensional, matching and strictly time ordered.
- Spectral analysis requires approximately uniform sampling.
- Coil area and turn count must be provided together for the ideal dB/dt estimate.
- Field integration requires a documented sign convention and defensible baseline; integration drift is not automatically corrected.
- Mode-number estimation requires at least three channels, finite toroidal angles in radians and adequate coherence. It is a candidate for expert review, not a standalone physical identification.
- No device-specific transfer function, calibration curve, pickup subtraction, equilibrium reconstruction or uncertainty propagation is inferred.

## Validation

The automated tests use synthetic sinusoidal signals, known coil parameters, a known linear field perturbation, invalid-input cases and a synthetic multi-channel phase pattern. Experimental validation still requires calibrated device data and documented coil geometry.
