# Execution contracts and scientific maturity

## Four distinct execution levels

| Level | Meaning | Permitted claim |
|---|---|---|
| `reference_only` | Scientific method and prerequisites are documented; no executable workflow is claimed. | Reference material only |
| `limited_calculation` | One or more numerical calculations exist for preprocessed inputs. | Limited calculation, not raw-instrument inference |
| `analysis_pipeline` | A declared sequence transforms supported raw or calibrated inputs into stated quantities. | Pipeline only within documented scope |
| `inverse_reconstruction` | An inverse problem uses a documented forward model, geometry/response and reconstruction settings. | Reconstruction only for the stated model and calibration |

A function existing in the repository does not establish an end-to-end instrument workflow. A successful numerical fit does not prove that physical assumptions are satisfied.

## Contract schema

Every row in `DIAGNOSTIC_REGISTRY.json` now has an `execution_contract` with execution level, completeness, input/output specs, assumptions, preprocessing, algorithm, uncertainty model, limitations, function links and a raw-data-to-physics flag.

The first migration deliberately marks contracts `incomplete` until input/output schemas, units, uncertainty handling and assumptions are actually specified. Empty specifications mean "not declared", not "no inputs required". UI code must show that state rather than inventing defaults. The typed adapter in `tis.contracts` reads the same JSON records. The separate 80-entry callable registry also carries conservative contracts; callable resolution alone is not treated as a complete raw-data workflow. Contract completeness is not experimental validation.

`FORMULA_REGISTRY.json` separately identifies formula-to-code links as `not_mapped` until an implementation and supporting tests are explicitly associated.

## Canonical data model

`tis.data_model.Measurement` carries quantity, unit, shot ID, timebase, location, coordinate system, sample rate, calibration, provenance, uncertainty, quality flags and invalid-result reason. Unit conversion is explicit and uses Pint.

## Reproducible projects

`tis.project` defines portable project and analysis-run records. Each run stores input references, settings, algorithm version, outputs, units, warnings and status. Project JSON is atomically written and integrity-checked with SHA-256. Large raw signals should remain in native files; record stable paths, checksums and acquisition metadata in `input_provenance`.

## Migration boundary

The new contract/data-model/project modules are the first migration boundary. Existing `app.py` and `tis/core.py` remain compatibility entry points during incremental page and numerical-domain extraction. Do not remove existing functions until public callers and tests are migrated.
