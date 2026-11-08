# Community analysis contributions

Tokamak Insight Suite can export a portable JSON contribution record for an analysis
case. The record captures the analysis identity, software version, parameters,
dataset SHA-256 (when available), optional shot/diagnostic/units, and result. A
reference result may be included when readily available, but is not required. It
deliberately does **not** embed the raw experimental file by default.

## Recommended contribution workflow

1. Keep the original raw data and source metadata locally.
2. Run an analysis and record units, calibration, preprocessing, assumptions, and
   any known quality flags.
3. Export a contribution record and inspect it for confidential or identifying
   metadata before sharing.
4. Share a public dataset URL, uncertainty information, and enough steps to reproduce
   the result when possible. A published or independently processed reference is optional.
5. Do not present a community contribution as a universal or device-independent result;
   report the data, assumptions, and known limits that define its scope.

The contribution schema is a transport format, not an automatic upload service or
scientific endorsement. Independent scientific validation is not required to use or
contribute to TIS; community feedback can improve the implementation over time. Public deployment must not silently retain uploaded
experimental files. A future submission endpoint requires an explicit retention,
consent, moderation, and security policy.
