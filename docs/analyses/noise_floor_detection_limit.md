# Noise floor and detection limit

`tis.signals.noise_floor_detection_limit` estimates a robust noise scale and a per-sample detection threshold from a selected background/no-signal interval.

## Example

```python
from tis.signals import noise_floor_detection_limit

result = noise_floor_detection_limit(
    background_samples,
    false_alarm_probability=1e-3,
    two_sided=False,
)
```

## Reported quantities

- `baseline`: median background level.
- `noise_floor_sigma`: robust scale estimated as `1.4826 × MAD`, with a standard-deviation fallback if the robust scale is numerically degenerate.
- `empirical_detection_threshold_above_baseline`: residual quantile for the selected tail probability.
- `gaussian_reference_threshold_above_baseline`: Gaussian reference threshold, reported for comparison rather than assumed to be the empirical distribution.
- `noise_estimate_degenerate`: indicates whether the robust scale required the fallback.

## Interpretation limits

Use samples that genuinely represent background/no-signal conditions and retain their original detector units. `false_alarm_probability` is a *per-sample* tail probability. This routine does not infer acquisition bandwidth, temporal correlation, averaging, family-wise false-alarm probability, instrument response, calibration or a device-specific physical detection limit. The Gaussian threshold is only a reference; prefer the empirical threshold when the observed background distribution is not well described by Gaussian noise.

This calculation is a software utility, not independent experimental validation.
