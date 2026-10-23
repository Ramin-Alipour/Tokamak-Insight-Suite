# Implementation coverage and scientific validation status

**Status snapshot:** 2026-10-23  
**Scope:** the main-branch registries and the executable/validation contracts they currently declare.

## Diagnostic coverage is not 80 complete analyses

The canonical `DIAGNOSTIC_REGISTRY.json` contains 80 diagnostic entries. Its implementation-status labels currently classify them as:

| Registry status | Count | What this means |
|---|---:|---|
| `Implemented numerical workflow (see limits)` | 20 | Some executable calculations exist, with the limitations in each contract; this is not a claim of complete raw-instrument-to-physics analysis. |
| `Registry reference only` | 59 | The catalog entry is not evidence of an implemented, callable end-to-end analysis. |
| Thomson scattering: scoped workflows; overall analysis incomplete | 1 | Several limited workflows are linked, but the overall diagnostic remains explicitly incomplete. |

All 80 entries are conservatively marked `completeness: incomplete` in their execution contracts. Do not advertise the registry as 80 complete analyses. A Python module existing somewhere in the repository does not promote an entry to implemented or validated.

## Formula coverage is also deliberately unverified

`FORMULA_REGISTRY.json` currently contains 37 formula references. All 37 have `implementation_link_status: not_mapped`, empty input/output specifications, and no linked tests. This is a known coverage gap, not proof that no related code exists. Until a formula has a specific callable mapping, explicit input/output units and preconditions, documented assumptions, and a test against an independent analytic or reference value, the catalog must not claim that formula is implemented or validated.

## Required completion gate for each analysis

An entry may be promoted only when all applicable gates below are reviewed and represented in version-controlled metadata:

1. **Executable mapping:** every linked module and function imports and resolves to a callable.
2. **Input contract:** shape, finite-value policy, units, calibration requirements, geometry, instrument response and valid operating range are explicit.
3. **Physical model:** equation and primary reference are documented; assumptions and failure conditions are surfaced to users.
4. **Output contract:** names, units, sign conventions, invalid-result behavior and uncertainty model are explicit.
5. **Numerical tests:** analytic solutions, dimensional checks, boundary cases, uncertainty checks and regression tests cover the implementation.
6. **Empirical validation where claimed:** the raw/processed status of the source data is documented; the reference is independent and measures the same quantity under compatible conventions; numerical discrepancy and uncertainty are recorded; and a reproducible regression fixture is archived. A plot, a synthetic fixture, or a value produced by the same implementation is not independent empirical validation.
7. **User-facing truthfulness:** the UI, CLI, catalog and reports expose incomplete status and caveats rather than implying a complete diagnostic reconstruction.

## Empirical validation policy

The existing [external empirical validation protocol](external_empirical_validation_protocol.md) remains the acceptance authority for experimental comparisons. MAST integration verifies the explicitly documented data-import and signal-processing paths; it does not validate all 80 entries. A GOLEM retrieval/format failure is unresolved, not a successful scientific comparison. Digitized points from a published plot must never be described as raw detector data.

## Release gate

A release must not be described as fully verified while required CI jobs fail, formula mappings remain unverified, or an analysis lacks the validation level claimed by its user-facing label. Record the exact commit, dependency/runtime versions, test results, source-data provenance and independent reference for each promotion.
