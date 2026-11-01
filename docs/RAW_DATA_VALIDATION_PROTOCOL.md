# Calibrated-Input and Software-Verification Contract

## Binding input assumption

Every diagnostic analysis receives measured data from the instrument that are already prepared, calibrated, ordered and suitable for the selected analysis. The software's scientific task is to infer physical parameters from those measured data using documented physical models.

The software does **not**:
- calibrate an instrument or detector;
- correct raw counts, detector response, wavelength axes, gain, offsets or acquisition timing;
- repair, reorder, normalize or otherwise prepare the supplied measurements;
- infer an unprovided instrument response or silently invent geometry, species composition, boundary conditions or other physical metadata;
- perform or claim experimental validation as part of the current development scope.

A measured response operator or other fixed instrument metadata may be supplied by the researcher and used as part of a forward model. It is not estimated or calibrated by the analysis unless a separate, explicit feature request authorizes that work.

## Physical-result classification

Every physical output must be placed in exactly one of these categories:

1. **Primary** — uniquely identifiable from the supplied measured data and explicitly supplied fixed model metadata. Compute automatically; do not ask the researcher to provide an initial guess as though it were a measurement.
2. **Conditional** — potentially identifiable only after named auxiliary inputs are supplied. Report the missing inputs, the physical reason they are needed, and the conditional output that becomes available.
3. **Non-identifiable** — not uniquely recoverable from the supplied measurement geometry and selected physical model. Do not report a fabricated, regularization-selected or optimizer-selected value as a measured result.

Numerical regularization, optimizer initialization, bounds and statistical priors are computational choices, not measured physical parameters. Any output that depends on them must expose that dependence.

## Uncertainty and provenance

- Report units for every dimensional physical parameter.
- Propagate supplied measurement uncertainty/covariance and supplied auxiliary-input uncertainty where implemented.
- State when uncertainty is unavailable, conditional, local, approximate, or excludes unprovided correlations/model discrepancy.
- Preserve the distinction between measured quantities, inferred physical parameters, numerical representations and non-identifiable quantities.
- Synthetic and unit tests verify software behavior only. They are not experimental validation.

## Software verification policy

The development gate is software verification: independent unit tests for physical formulas, forward models, inverse routines, uncertainty propagation, identifiability, API contracts, registry synchronization and user-interface integration. GitHub Actions should run the relevant software tests on changes to the main branch.

No experimental validation, benchmark reproduction against measured-device reference results, or agreement with published experimental outcomes is required or claimed during this development phase. Such work can only be introduced under a separate explicit authorization and must not be silently mixed into feature implementation.

## Thomson Analysis 1

For Optical Thomson Scattering, Thomson Spectral Distribution, Thomson Spatial Profile Reconstruction and Collective Thomson Scattering (electrostatic and full-EM), use the same calibrated measured-input contract and the same Primary / Conditional / Non-identifiable classification. Optical Thomson and CTS remain physically distinct model branches. A single scattering geometry cannot identify an unrestricted three-dimensional electron velocity distribution; a projected one-dimensional marginal must be labelled as such.

The physical model shortlist and literature provenance are maintained in `docs/PHYSICS_MODEL_SELECTION_POLICY.md`. The engineering scope for audit items 7–110 is maintained in `docs/analyses/thomson_scattering_gap_closure_plan.md`.
