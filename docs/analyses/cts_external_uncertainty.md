# Full-EM CTS external-input uncertainty contract

## Implemented local-input propagation

The local external-input covariance route is connected to
`tokamak_analysis.cts_full_em.analyze_full_em_cts_spectrum` and exposed in
the Full-EM CTS dashboard. Supply both:

- `external_input_covariance`: a finite symmetric positive-semidefinite
  covariance matrix;
- `external_input_parameter_order`: the exact ordered names corresponding
  to its rows and columns.

Supported local components are magnetic-field Cartesian components,
incident/scattered direction components, real/imaginary components of the
incident/scattered complex polarization vectors, and indexed ion composition
terms (`ion_fraction_i`, `ion_mass_amu_i`, `ion_charge_state_i`). Names must
be unique. The API samples the joint covariance with a reproducible seed and
refits the complete Full-EM inverse for each draw. Successful refit parameter
vectors are used to estimate the local-external-input contribution to the
fitted-parameter covariance.

The result reports the supplied parameter order, requested/successful draw
counts, whether propagation met its success threshold, and the separate
`local_external_inputs` covariance component. The conditional fit covariance
is retained separately; when propagation succeeds, the reported total fit
covariance is the sum of the conditional fit and the propagated local-input
component. This Monte-Carlo contribution is distinct from spectral
measurement-noise uncertainty.

## Strict limits (not implemented by this route)

- This local-vector contract rejects frequency-resolved ray/equilibrium
  geometry. Use the separate
  `propagate_full_em_cts_ray_geometry_uncertainty` workflow with a sampler
  that regenerates the frequency-resolved geometry and performs full-forward
  refits. Its uncertainty contribution is a separate covariance component.
- Cross-covariance between the local-vector contract and independently
  supplied ray/equilibrium, instrument-response, or other external covariance
  sources is not represented by simply adding their separate covariance
  matrices. A joint sampler must perturb correlated inputs in the same
  forward refit to retain such cross terms.
- The local covariance route does not model diffraction, non-paraxial beam
  propagation, full vector mode conversion, or unimplemented kinetic effects.
- Invalid/failed Monte-Carlo draws are not physical measurements. If too few
  refits succeed, the result must report propagation as unsuccessful and must
  not add a fabricated covariance contribution.

No instrument calibration or experimental validation is performed. All
spectra and response operators supplied to this analysis are assumed to be
already measured and calibrated.

## Verification coverage

The repository tests cover covariance validation (including singular positive
semidefinite matrices and invalid contracts), reproducible sampling, direct
cross-covariance propagation, the Full-EM inversion integration and its
failure/domain reporting, and separate ray-geometry refitting. These are
software regression tests, not experimental validation.
