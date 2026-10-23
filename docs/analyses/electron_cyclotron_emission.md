# Electron Cyclotron Emission (ECE) — analysis 3/80

## Scope and status

The ECE workflow is implemented in `tokamak_analysis/ece_analysis.py` and exposed in the Streamlit workspace. It supports three explicitly scoped workflows:

1. **Brightness temperature** from absolutely calibrated spectral radiance.
2. **Channel fluctuations and one-sided periodogram PSD** from aligned multi-channel time series.
3. **Cold resonance mapping** from channel frequency to resonance field, with an optional simplified circular toroidal-field major-radius estimate.

The implementation is complete within this scope. It is not a general-purpose ECE radiative-transfer solver, equilibrium mapper, or instrument-calibration package. Numerical/software verification does not constitute experimental validation for a particular tokamak.

## Physics relations

### Cyclotron resonance

For non-relativistic electrons at harmonic \(s\),

\[
f_{ce}=\frac{e|B|}{2\pi m_e},\qquad
f=s f_{ce},\qquad
B_{res}=\frac{2\pi m_e f}{s e}.
\]

The implementation is `resonance_field_T(frequency_Hz, harmonic)`. Frequencies must be finite and positive; the harmonic must be a positive integer. Relativistic mass shifts, Doppler shifts, finite-orbit effects, and detailed polarization/cutoff effects are not modeled.

### Simplified major-radius mapping

For a circular, axisymmetric, toroidal-field-only approximation,

\[
B_\phi(R)=B_0\frac{R_0}{R},\qquad
R_{res}=R_0\frac{B_0}{B_{res}}.
\]

The implementation is `resonance_radius_m(frequency_Hz, major_radius_m, B_axis_T)`. This is a rough coordinate estimate, not an EFIT-equilibrium ray trace. Frequency alone cannot identify which side of the plasma a line of sight intersects; a mirrored high-field-side radius is deliberately not fabricated. Shaping, poloidal field, finite channel bandwidth, refraction, relativistic/Doppler shifts, and viewing geometry may materially affect localization.

### Rayleigh–Jeans brightness temperature

For calibrated spectral radiance (I_\nu),

\[
T_b=\frac{c^2 I_\nu}{2 k_B\nu^2},\qquad
T_b[\mathrm{eV}]=\frac{k_B T_b[\mathrm K]}{e}.
\]

The implementation is `brightness_temperature_K` and `brightness_temperature_eV`. Input radiance must be finite and non-negative and frequency finite and positive. The input must be a calibrated spectral radiance in W m⁻² sr⁻¹ Hz⁻¹, not raw detector counts, arbitrary voltage, or an uncalibrated channel amplitude.

**Brightness temperature is not automatically electron temperature.** Interpreting \(T_b\) as \(T_e\) requires optically thick emission and a thermalized electron distribution, in addition to valid calibration and appropriate propagation assumptions. The optional `optical_depth` argument is recorded as metadata only: the code does not infer thermalization or certify that a channel is optically thick from optical depth alone.

### Channel fluctuation spectra

For a sampled channel signal \(x_j(t)\), the workflow subtracts each channel's temporal mean and computes an FFT-based one-sided periodogram. It reports per-channel fluctuation RMS, PSD, sample frequency, and Nyquist frequency. The periodogram normalization is checked against Parseval's relation for odd and even sample counts. DC and (for even sample counts) Nyquist bins are not doubled.

Input time samples must be finite, strictly increasing, and approximately uniformly spaced (relative tolerance 1e-3 against the median step); all signal samples must be finite. This is not an automatic anti-aliasing filter or a substitute for checking acquisition electronics, timing jitter, channel synchronization, gaps, calibration, and noise. The optional `noise_std` is a supplied per-channel RMS noise estimate; the code only reports whether observed RMS exceeds that estimate and does not perform a frequency-dependent noise-floor subtraction.

## API

```python
from tokamak_analysis.ece_analysis import analyze_ece

# Brightness temperature from calibrated radiance
result = analyze_ece(
    [80e9, 90e9, 100e9],
    harmonic=2,
    spectral_radiance_W_m2_sr_Hz=[1e-10, 1.2e-10, 1.1e-10],
    mode="brightness_temperature",
)

# Resonance field and approximate major-radius coordinate
mapping = analyze_ece(
    [80e9, 90e9, 100e9],
    harmonic=2,
    major_radius_m=1.0,
    B_axis_T=2.0,
    mode="resonance_mapping",
)

# Multi-channel fluctuation RMS and PSD
import numpy as np
time_s = np.arange(4096) * 1e-6
signal = np.column_stack([
    np.sin(2 * np.pi * 10e3 * time_s),
    np.sin(2 * np.pi * 20e3 * time_s),
])
fluctuations = analyze_ece(
    [80e9, 90e9],
    harmonic=2,
    signal=signal,
    time_s=time_s,
    mode="channel_fluctuations",
)
```

### Modes and inputs

| Mode | Required inputs | Principal outputs |
|---|---|---|
| `brightness_temperature` | `frequency_Hz`, `spectral_radiance_W_m2_sr_Hz` | `resonance_field_T`, `brightness_temperature_K`, `brightness_temperature_eV` |
| `channel_fluctuations` | `frequency_Hz`, `time_s`, `signal` with shape (time samples, channels) | `fluctuation_rms`, `fluctuation_psd_frequency_Hz`, `fluctuation_psd`, sample and Nyquist frequencies |
| `resonance_mapping` | `frequency_Hz`; optional paired `major_radius_m` and `B_axis_T` | `resonance_field_T`, optional `resonance_radius_m` |

All channel center frequencies must be a non-empty, finite, positive, strictly increasing one-dimensional array. `harmonic` must be a positive integer. Mode-specific conflicting inputs are rejected. Optional `channel_bandwidth_Hz` must be positive and scalar or one value per channel; its fractional value is reported but no finite-bandwidth convolution is performed.

## Validation

Automated tests are in `tests/test_ece_analysis.py`. They cover:
- fundamental and harmonic resonance scaling;
- inversion of the simplified toroidal-field model;
- zero-radiance brightness temperature and required calibrated radiance;
- multi-time/multi-channel radiance shapes;
- RMS and PSD outputs, sampling frequency and Nyquist frequency;
- Parseval consistency for odd and even sample counts, including correct Nyquist-bin normalization;
- rejection of invalid frequencies, harmonics, radiance, sampling intervals, mode-conflicting inputs, and malformed noise/optical-depth arrays;
- refusal to fabricate a high-field-side radius without viewing geometry.

These tests establish analytic/software behavior for the stated model. They do **not** establish calibration traceability, instrument response, optical depth, equilibrium accuracy, or agreement with independent experimental reference data.

## Scientific acceptance checklist for a real discharge

Before interpreting results, document and verify:

- channel center-frequency calibration, bandwidth, polarization, and receiver response;
- absolute radiometric calibration and its uncertainty for brightness-temperature work;
- the optical-depth and thermalization basis before treating brightness temperature as electron temperature;
- equilibrium and viewing geometry, resonance layer, harmonics, relativistic/Doppler effects, and cutoff/refraction for spatial interpretation;
- sample timing, synchronization, missing samples, anti-aliasing, signal units, and acquisition bandwidth for fluctuation spectra;
- noise-floor estimate, channel-to-channel calibration, and reproducibility across independent analysis settings;
- comparison with an independent diagnostic or a trusted synthetic diagnostic before claiming device-level validation.

The workflow intentionally reports the model and its limitations rather than silently upgrading a simplified estimate into a device-validated physical inference.
