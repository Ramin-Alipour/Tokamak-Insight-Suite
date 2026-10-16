# Community analysis contributions

Tokamak Insight Suite can export a portable JSON contribution record for an analysis
case. The record captures the analysis identity, software version, parameters,
dataset SHA-256 (when available), optional shot/diagnostic/units, result, and an
optional reference result. It deliberately does **not** embed the raw experimental
file by default.

## Recommended contribution workflow

1. Keep the original raw data and source metadata locally.
2. Run an analysis and record units, calibration, preprocessing, assumptions, and
   any known quality flags.
3. Export a contribution record and inspect it for confidential or identifying
   metadata before sharing.
4. When possible, provide a public dataset URL, a reference result derived from the
   same shot/data, uncertainty information, and enough steps to reproduce the result.
5. Mark results as independently verified only after a separate reviewer reproduces
   the comparison. Exported records are explicitly marked as not independently verified.

The contribution schema is a transport format, not an automatic upload service or
scientific validation claim. Public deployment must not silently retain uploaded
experimental files. A future submission endpoint requires an explicit retention,
consent, moderation, and security policy.
