# Reference-result validation protocol

The goal is not merely to produce an output; it is to establish whether a TIS result reproduces a defensible published or independently measured result.

## Reference-data priority and conflicts

1. Prefer the original diagnostic data and the authors' acquisition/calibration metadata whenever available.
2. When original raw data are unavailable, digitize the relevant published figure and record the article DOI, figure/panel, axes/units, digitization method and estimated digitization uncertainty. A digitized curve is a reference result, **not raw diagnostic input**.
3. Record numerical values stated in the article text separately. If text and plotted values disagree, do not silently merge them: retain both references, identify the discrepancy, and use the graph-derived value as the provisional comparison target only when its axes and plotted series can be identified unambiguously. State this choice in the comparison notes.
4. Never describe synthetic data generated from a formula as experimental validation.

## What must match before comparing

- Same physical quantity and definition (measured, derived or inferred).
- Same units and conversion conventions.
- Same shot/discharge, time interval, spatial location and diagnostic channel where relevant.
- Same calibration, filtering, detrending, windowing, sampling rate and spectral normalization.
- Same model assumptions and fit interval.
- Uncertainty from the paper/data source plus digitization and TIS fit/measurement uncertainty where applicable.

## Use the TIS comparison workflow

Open **Reference Results Validation** and upload a CSV with:
- Required: `reference_value`, `software_value`
- Recommended: `parameter`, `unit`, `source`
- If available: `reference_uncertainty`, `software_uncertainty`
- Otherwise: an explicit `absolute_tolerance` or `relative_tolerance`

The workflow calculates signed and absolute differences. If both standard uncertainties are supplied, it compares the absolute difference with (k\sqrt{u_{ref}^2+u_{TIS}^2}). Otherwise it uses an explicitly supplied tolerance. If neither is supplied, the row is marked **UNASSESSED**; the software does not invent a pass/fail threshold.

## Interpretation

- **AGREES** means only that the numbers agree under the supplied uncertainty/tolerance rule.
- **DISAGREES** means the difference exceeds that rule and needs investigation.
- **UNASSESSED** means no defensible decision criterion was supplied.
- Agreement does not independently validate the diagnostic calibration, physical model, source data, or a whole paper's conclusions.

For every benchmark, preserve the original source, data-file SHA-256, analysis settings, software version, output and comparison table. Keep test-only reference/digitized datasets separate from the shipped application and do not present them as user measurements.
