# Diagnostic status report

The canonical diagnostic catalogue and analysis registry each contain 80 entries, with matching IDs, titles, and order. The analysis registry points to software entry points; that mapping alone does not imply full physical modeling or independent validation.

Generate a reconciled JSON report from the same source used by the catalogue UI:

```bash
python -m tis.cli catalog --kind status --output-dir artifacts/diagnostic-status
```

This writes `DIAGNOSTIC_STATUS_REPORT.json` with:
- the number of diagnostic records and their execution-level/completeness counts;
- the callable entry-point count and its completeness counts, separately;
- a per-record view of the first ten diagnostics;
- explicit interpretation notes that distinguish software-scope completeness from independent experimental validation.

For the current main-branch registry, the expected diagnostic-level counts are 80 records: 52 `reference_only`, 17 `limited_calculation`, and 11 `analysis_pipeline`. Contract completeness is 69 `incomplete`, 7 `complete`, and 4 `complete_with_declared_scope`. These are registry metadata counts, not claims that 10 diagnostics are independently validated. The current canonical registry must report exactly `complete: 7`, `complete_with_declared_scope: 4`, and `incomplete: 69`. The generated report now includes software version, canonical registry path and SHA-256, and explicit count-consistency booleans; retain these fields when comparing output across deployments. If a dashboard, downloaded JSON, or older report says all 80 contracts are incomplete (or shows the older 59/20/1 execution split), it is stale or was generated from a different checkout; regenerate the report using the command above and verify the reported source commit before interpreting it. In particular, Thomson scattering is currently `complete_with_declared_scope` in the canonical registry: its declared software workflows are implemented, while independent experimental validation and device-specific calibration remain outside that software-completion label. Do not describe it as registry-incomplete. A yellow status in an external/manual review should be labeled as an empirical-validation caveat, not as missing implementation, unless a concrete software defect is identified.

Do not manually force a record to `complete` merely to make a dashboard green. Change completeness only after the applicable implementation contract, numerical tests, documented assumptions/limitations, and any validation claims are supported by evidence. A callable entry point or passing synthetic test alone does not establish independent experimental validation.
