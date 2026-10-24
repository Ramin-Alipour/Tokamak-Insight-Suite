# Implementation coverage and scientific validation status

**Status snapshot:** 2026-10-24  
**Scope:** the main-branch registries and the executable/validation contracts they declare.

## Diagnostic categories versus callable analysis entry points

The canonical `DIAGNOSTIC_REGISTRY.json` contains 80 diagnostic records, while `tokamak_analysis.analysis_registry.ANALYSES` contains 81 callable entry points. These registries describe different layers and are not a one-to-one row mapping: one diagnostic may expose multiple callables, and a callable entry point is not itself a new diagnostic. The CI contract therefore verifies the two counts independently and resolves every callable by ID.

The 80 diagnostic contracts currently classify execution level as:

| Execution level | Count | What this means |
|---|---:|---|
| `reference_only` | 52 | The catalog entry is not evidence of an implemented callable numerical workflow. |
| `limited_calculation` | 17 | A scoped numerical calculation exists, with assumptions and limitations. |
| `analysis_pipeline` | 11 | A multi-step executable workflow exists; this does not imply independent experimental validation. |

Contract completeness is currently `incomplete` for 69 diagnostics, `complete` for 8 within the declared computational scope, and `complete_with_declared_scope` for 3. This count must be generated from the canonical registry; do not copy it from a stale dashboard or report. These are software-scope labels only. They do **not** certify device-specific calibration, agreement with experimental references, or independent scientific validation. Thomson scattering is implemented within its declared computational scope; device-specific calibration and independent experimental comparisons remain recommended research responsibilities, not release/completion gates under the open research beta policy.

A provisional real-data cross-diagnostic consistency check for MAST shot 28830 passed the declared 10% tolerance against rounded published Table II values: edge density relative difference −3.77% and interferometer line-density relative difference −1.74%. The reference does not specify the exact averaging interval, and the input consists of archived Thomson-derived profiles plus an independent interferometer signal. This is a useful external consistency check, not validation of the raw-spectrum fit or a full Thomson spectrum-to-density inference chain.

A Python module existing somewhere in the repository does not by itself promote a diagnostic to implemented or experimentally validated.

## Formula coverage: 37/37 callable mappings

The canonical `FORMULA_REGISTRY.json` now maps all 37 formula entries to a named callable in `tokamak_analysis.formula_library`. Every entry declares its input and output quantities/units, preconditions, uncertainty policy and a linked analytic reference test. The packaged copy at `tis/data/FORMULA_REGISTRY.json` is kept byte-for-byte semantically synchronized with the root registry.

This mapping is a contract-level milestone, **not a claim that all 37 formulas have been empirically validated on tokamak measurements**. Analytic unit/reference tests establish software-level correctness for the stated mathematical model. They do not establish that the model's assumptions, calibration, geometry, instrument response, time window or data provenance are valid for a particular experiment.

Some entries are intentionally limited calculations:
- Doppler temperature accepts an already corrected thermal Gaussian sigma; it does not infer or subtract instrument/Stark/Zeeman broadening.
- The safety-factor formula is a large-aspect-ratio estimate, not equilibrium reconstruction.
- Ion-saturation current-to-flux is a model-dependent proxy.
- The bolometry routine is an unconstrained linear Tikhonov estimate; it does not enforce non-negative emissivity or validate the geometry matrix.
- Coherence and cross-correlation helpers do not infer causality or confidence intervals.
- Greenwald density is an empirical reference scaling, not a universal operational limit.

## Required completion gate for each analysis

An entry may be promoted only when all applicable gates below are reviewed and represented in version-controlled metadata:

1. **Executable mapping:** every linked module and function imports and resolves to a callable.
2. **Input contract:** shape, finite-value policy, units, calibration requirements, geometry, instrument response and valid operating range are explicit.
3. **Physical model:** equation and primary reference are documented; assumptions and failure conditions are surfaced to users.
4. **Output contract:** names, units, sign conventions, invalid-result behavior and uncertainty model are explicit.
5. **Numerical tests:** analytic solutions, dimensional checks, boundary cases, uncertainty checks and regression tests cover the implementation.
6. **Empirical validation where claimed:** the raw/processed status of source data is documented; the reference is independent and measures the same quantity under compatible conventions; numerical discrepancy and uncertainty are recorded; and a reproducible regression fixture is archived. A plot, a synthetic fixture, or a value produced by the same implementation is not independent empirical validation.
7. **User-facing truthfulness:** the UI, CLI, catalog and reports expose incomplete status and caveats rather than implying a complete diagnostic reconstruction.

## Empirical validation policy

The existing [external empirical validation protocol](external_empirical_validation_protocol.md) remains the acceptance authority for experimental comparisons. MAST integration verifies the explicitly documented data-import and signal-processing paths; it does not validate all 80 entries. A GOLEM retrieval/format failure is unresolved, not a successful scientific comparison. Digitized points from a published plot must never be described as raw detector data.

## Release gate

A release must not be described as fully verified while required CI jobs fail, an analysis lacks the validation level claimed by its user-facing label, or empirical validation is claimed without an independent compatible reference. Record the exact commit, dependency/runtime versions, test results, source-data provenance and independent reference for each promotion.
