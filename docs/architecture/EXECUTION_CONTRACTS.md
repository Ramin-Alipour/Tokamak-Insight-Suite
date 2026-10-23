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

`FORMULA_REGISTRY.json` separately maps all 37 catalogue formulae to callable functions, declared input/output units, preconditions, uncertainty policies and analytic reference tests. These are software/analytic verification contracts, not empirical validation on tokamak measurements. The root registry and packaged copy must remain synchronized; `tests/test_formula_registry_complete.py` enforces this mapping gate.

## Canonical data model

`tis.data_model.Measurement` carries quantity, unit, shot ID, timebase, location, coordinate system, sample rate, calibration, provenance, uncertainty, quality flags and invalid-result reason. Unit conversion is explicit and uses Pint.

## Reproducible projects

`tis.project` defines portable project and analysis-run records. Each run stores input references, settings, algorithm version, outputs, units, warnings and status. Project JSON is atomically written and integrity-checked with SHA-256. Large raw signals should remain in native files; record stable paths, checksums and acquisition metadata in `input_provenance`.

## Migration boundary

The new contract/data-model/project modules are the first migration boundary. Existing `app.py` and `tis/core.py` remain compatibility entry points during incremental page and numerical-domain extraction. Do not remove existing functions until public callers and tests are migrated.


## Measurement validation and unit-conversion rules

The canonical Measurement object validates declared quantity/unit, metadata mapping types, uncertainty shape and sign, coordinate finiteness, timebase monotonicity, and consistency between an explicitly supplied sample rate and the median time interval (2% tolerance). For irregularly sampled data, omit sample_rate_hz and preserve the explicit time coordinates. Non-finite measurement samples are intentionally retained to represent acquisition gaps; producers should attach quality flags or a non-valid status with a reason.

Unit conversion treats uncertainty as a scale-only quantity. For affine units such as degrees Celsius and kelvin, the value receives the offset while the standard uncertainty does not. Regression tests cover this distinction. This conversion rule does not infer calibration uncertainty or instrument response.

## Reproducible analysis-run records

AnalysisRun supports explicit software_version, environment metadata, input_hashes (SHA-256 hex digests keyed by input name), and random_seed. Its reproducibility_fingerprint hashes the analysis name, input references, settings, algorithm/software versions, environment, input hashes, declared units and random seed. output_digest hashes normalized outputs separately. The fingerprint records configuration; it does not guarantee bitwise-identical results across platforms or establish scientific validity. Callers must populate meaningful version/environment fields and calculate hashes from the actual input bytes.
