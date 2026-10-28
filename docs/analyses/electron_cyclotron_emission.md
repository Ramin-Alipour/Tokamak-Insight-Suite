# Electron Cyclotron Emission (ECE) — analysis 3/80

## Scope and status

The ECE workflow is implemented in `tokamak_analysis/ece_analysis.py` and exposed in the Streamlit workspace. It supports five explicitly scoped workflows:

1. **Brightness temperature** from absolutely calibrated spectral radiance.
2. **Channel fluctuations, full-record one-sided periodogram, and Hann-window Welch PSD** from aligned multi-channel time series, with approximate PSD confidence intervals and an optional white-noise reference.
3. **Cold resonance mapping** from channel frequency to resonance field, with an optional simplified circular toroidal-field major-radius estimate.
4. **Receiver-response convolution** of an already-calibrated spectrum with a user-supplied, validated channel response kernel, using in-range interpolation and trapezoidal integration.
5. **Time-windowed radial electron-temperature profile reconstruction** from already-calibrated brightness-temperature samples, using supplied optical depth with a single-slab LTE inversion or an explicit optically-thick assumption. Optional profile mapping supports a supplied monotonic toroidal-field table or externally derived channel resonance locations.

The dashboard exposes all five workflows. It is not a general-purpose ECE radiative-transfer solver, equilibrium mapper, or instrument-calibration package. Numerical/software verification does not establish experimental validation for a particular tokamak.

## Physics relations

### Cyclotron resonance

For non-relativistic electrons at harmonic \(s\),

\[
f_{ce}=\frac{e|B|}{2\pi m_e},\qquad
f=s f_{ce},\qquad
B_{res}=\frac{2\pi m_e f}{s e}.
\]

The implementation is `resonance_field_T(frequency_Hz, harmonic)`. Frequencies must be finite and positive; the harmonic must be a positive integer. The default is cold/non-relativistic. `resonance_field_relativistic_T` optionally applies a monoenergetic relativistic/Doppler sensitivity correction using user-supplied kinetic energy and signed line-of-sight beta. This is not a distribution-averaged emissivity solver. Finite-orbit effects, polarization/cutoff effects and full ray tracing remain outside scope.

### Simplified major-radius mapping

For a circular, axisymmetric, toroidal-field-only approximation,

\[
B_\phi(R)=B_0\frac{R_0}{R},\qquad
R_{res}=R_0\frac{B_0}{B_{res}}.
\]

The implementation is `resonance_radius_m(frequency_Hz, major_radius_m, B_axis_T)`. This is a rough coordinate estimate, not an EFIT-equilibrium ray trace. Frequency alone cannot identify which side of the plasma a line of sight intersects; a mirrored high-field-side radius is deliberately not fabricated. Shaping, poloidal field, finite channel bandwidth, refraction, relativistic/Doppler shifts, and viewing geometry may materially affect localization. The profile workflow reports a channel-bandwidth-derived radial localization envelope; it does not convolve a full receiver response into a radiative-transfer forward model.

### Rayleigh–Jeans brightness temperature

For calibrated spectral radiance (I_\nu),

\[
T_b=\frac{c^2 I_\nu}{2 k_B\nu^2},\qquad
T_b[\mathrm{eV}]=\frac{k_B T_b[\mathrm K]}{e}.
\]

The implementation is `brightness_temperature_K` and `brightness_temperature_eV`. Input radiance must be finite and non-negative and frequency finite and positive. The input must be a calibrated spectral radiance in W m⁻² sr⁻¹ Hz⁻¹, not raw detector counts, arbitrary voltage, or an uncalibrated channel amplitude.

**Brightness temperature is not automatically electron temperature.** Interpreting \(T_b\) as \(T_e\) requires optically thick emission and a thermalized electron distribution, in addition to valid calibration and appropriate propagation assumptions. In `analyze_ece`, optional `optical_depth` is metadata/conditioning information only; it does not certify thermalization. The separate profile workflow uses optical depth in an explicitly declared single-slab LTE inversion, or requires the user to affirm the optically thick approximation.

### Receiver-response convolution

`convolve_ece_receiver_response` integrates calibrated spectral radiance against user-supplied non-negative channel response weights. The spectrum and response grids must be finite, positive, strictly increasing, and fully covered by the measured spectrum; extrapolation is rejected. Response rows are normalized by their trapezoidal integral. This operation is not detector calibration, response-function inference, or radiative-transfer inversion. If channel brightness temperature is reported using the response-weighted center frequency, treat it as an approximation for broad channels.

## Channel fluctuation spectra

For a sampled channel signal \(x_j(t)\), the workflow subtracts each channel's temporal mean and computes an FFT-based one-sided periodogram. It reports per-channel fluctuation RMS, PSD, sample frequency, and Nyquist frequency. The periodogram normalization is checked against Parseval's relation for odd and even sample counts. DC and (for even sample counts) Nyquist bins are not doubled.

Input time samples must contain at least eight finite, strictly increasing, approximately uniformly spaced samples (relative tolerance 1e-3 against the median step); all signal samples must be finite. Records shorter than eight samples are rejected because the workflow always reports a Welch PSD and confidence bounds. The maximum relative sample-interval deviation is reported, but this cannot identify timing jitter below timebase resolution, repair acquisition gaps, validate synchronization, or provide anti-alias filtering. This is not an automatic anti-aliasing filter or a substitute for checking acquisition electronics, timing jitter, channel synchronization, gaps, calibration, and noise. The optional `noise_std` is a supplied per-channel time-domain RMS noise estimate. It reports RMS comparison and an expected one-sided PSD reference for white noise; this is not a measured noise floor and is not subtracted from the signal PSD.

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
| `channel_fluctuations` | `frequency_Hz`, `time_s`, `signal` with shape (time samples, channels); optional `noise_std`, `psd_confidence_level` | `fluctuation_rms`, full-record periodogram, Welch PSD, approximate Welch confidence bounds, expected white-noise PSD reference when noise RMS is supplied, sample and Nyquist frequencies, sampling-interval audit |
| `resonance_mapping` | `frequency_Hz`; optional paired `major_radius_m` and `B_axis_T` | `resonance_field_T`, optional `resonance_radius_m` |
| `receiver_response_convolution` | Calibrated spectrum, supplied spectrum/response frequency grids and response weights | Response-weighted channel radiance and effective center frequency; broad-channel Tb is approximate |
| `reconstruct_ece_temperature_profile` (separate workflow) | `frequency_Hz`, already-calibrated `calibrated_brightness_temperature_eV` matrix, `time_s`, `major_radius_m`, `B_axis_T`, and either per-channel `optical_depth` or explicit `assume_optically_thick=True`; optional background and time window | Time-window-averaged per-channel brightness temperature and inferred Te, approximate resonance radii, and interpolated radial Te profile; no extrapolation |

All channel center frequencies must be a non-empty, finite, positive, strictly increasing one-dimensional array. `harmonic` must be a positive integer. Mode-specific conflicting inputs are rejected. Optional `channel_bandwidth_Hz` must be positive and scalar or one value per channel; profile reconstruction uses it only for a radial localization envelope. The separate receiver-response workflow performs convolution only when a validated response kernel is supplied.

## Validation

Automated numerical tests are in `tests/test_ece_analysis.py`; Streamlit interaction coverage is in `tests/test_ece_dashboard.py`, with a dedicated real-browser smoke test in `tests/e2e/test_ece_live_browser.py`. They cover:
- fundamental and harmonic resonance scaling;
- inversion of the simplified toroidal-field model;
- zero-radiance brightness temperature and required calibrated radiance;
- multi-time/multi-channel radiance shapes;
- RMS and PSD outputs, sampling frequency and Nyquist frequency;
- Parseval consistency for odd and even sample counts, including correct Nyquist-bin normalization;
- rejection of invalid frequencies, harmonics, radiance, sampling intervals, mode-conflicting inputs, and malformed noise/optical-depth arrays;
- refusal to fabricate a high-field-side radius without viewing geometry.

The test suite also checks Welch confidence-bound ordering, the expected white-noise PSD reference, profile uncertainty propagation, bandwidth-derived radial envelopes, low-optical-depth conditioning warnings, optional time-resolved profiles, and dashboard wiring. These tests establish analytic/software behavior for the stated model. They do **not** establish calibration traceability, instrument response, optical depth, equilibrium accuracy, or agreement with independent experimental reference data.

## Scientific acceptance checklist for a real discharge

Before interpreting results, document and verify:

- channel center-frequency calibration, bandwidth, polarization, and receiver response;
- absolute radiometric calibration and its uncertainty for brightness-temperature work;
- the optical-depth and thermalization basis before treating brightness temperature as electron temperature;
- equilibrium and viewing geometry, resonance layer, harmonics, relativistic/Doppler effects, and cutoff/refraction for spatial interpretation;
- sample timing, synchronization, missing samples, anti-aliasing, signal units, and acquisition bandwidth for fluctuation spectra;
- noise-floor estimate, channel-to-channel calibration, and reproducibility across independent analysis settings;
- comparison with an independent diagnostic or a trusted synthetic diagnostic before claiming device-level validation.

Uncertainty estimates are first-order approximations. Optional channel covariance and temporal-autocorrelation adjustment are supported when supplied/selected; a user-supplied fractional model-discrepancy floor can be added, but unknown model discrepancy remains unquantified. The supplied channel covariance describes the time-window mean and is propagated through the mean radial profile; it is intentionally not applied to instantaneous time-resolved profiles. Use per-sample brightness-temperature uncertainties for time-resolved uncertainty estimates. These estimates do not include unprovided calibration, equilibrium or response-kernel uncertainty. Low optical depth can make the single-slab inversion ill-conditioned. The workflow intentionally reports the model and its limitations rather than silently upgrading a simplified estimate into a device-validated physical inference.

## Radial electron-temperature profile reconstruction

The dashboard also offers a **Temperature profile reconstruction** workflow for a time-by-channel table whose columns already contain calibrated brightness temperature in eV. It computes the time-window mean per channel and maps channel frequencies to approximate resonance radii using the same simplified toroidal-field model. The input values must already be calibrated before import; TIS does not apply gain/offset coefficients or perform detector calibration.

For temperature inference, the user must either supply optical depth per channel and a background brightness temperature, using the single-slab LTE relation

`Tb = Te (1 - exp(-tau)) + Tbg exp(-tau)`

or explicitly declare the optically thick approximation `Te ≈ Tb`. The app does not silently assume that brightness temperature is electron temperature. Optical depths below 1e-3 trigger a conditioning warning because inversion amplifies input errors approximately as 1/(1 − exp(−τ)).

The profile is a linear interpolation between distinct channel resonance locations and is restricted to the measured radial interval. An optional output retains inferred channel and radial profiles at each sample in the selected time window, rather than only the time-window mean. It is a profile estimate, not a regularized inversion of a full radiative-transfer forward model. Optional first-order uncertainty propagation accepts measurement/model-input uncertainties and channel covariance; temporal autocorrelation can adjust effective sample counts, and a user-selected fractional model-discrepancy floor can be added. Unknown covariance and model discrepancy remain unquantified. Finite channel bandwidth is represented as a radial localization envelope, not a receiver-response convolution. The workflow does not perform full ray tracing or reconstruct EFIT equilibrium. Results must therefore report these assumptions and must not be described as device-validated without independent validation.

