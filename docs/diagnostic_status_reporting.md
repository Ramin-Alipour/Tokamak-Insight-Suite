# Diagnostic status report

Use the diagnostic registry for counts of the 80 diagnostic catalogue records. The callable analysis registry is a different layer and currently contains 81 callable entry points; it must not be counted as though it were a one-to-one list of the 80 diagnostics.

Generate a reconciled JSON report from the same source used by the catalogue UI:

```bash
python -m tis.cli catalog --kind status --output-dir artifacts/diagnostic-status
```

This writes `DIAGNOSTIC_STATUS_REPORT.json` with:
- the number of diagnostic records and their execution-level/completeness counts;
- the callable entry-point count and its completeness counts, separately;
- a per-record view of the first ten diagnostics;
- explicit interpretation notes that distinguish software-scope completeness from independent experimental validation.

For the current main-branch registry, the expected diagnostic-level counts are 80 records: 52 `reference_only`, 17 `limited_calculation`, and 11 `analysis_pipeline`. Contract completeness is 70 `incomplete`, 8 `complete`, and 2 `complete_with_declared_scope`. These are registry metadata counts, not claims that 10 diagnostics are independently validated. In particular, Thomson Scattering remains conservatively marked incomplete pending independent experimental validation; Langmuir probe and the supported scoped workflows have their own declared implementation status.

Do not manually force a record to `complete` merely to make a dashboard green. Change completeness only after the applicable implementation contract, numerical tests, documented assumptions/limitations, and any validation claims are supported by evidence. A callable entry point or passing synthetic test alone does not establish independent experimental validation.
