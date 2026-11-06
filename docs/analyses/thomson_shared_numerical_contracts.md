# Analysis 1 shared numerical contracts

This module supplies reusable validation primitives for Analysis 1. It does not
claim to solve any missing kinetic equation or to make a model physically valid.

## Enforced contracts

- Omitted background metadata resolves to an uncharacterized offset, never to
  physical zero. A zero background is accepted only through the explicit
  `assumed_zero` declaration.
- Reference-PCA and measured-background states require the corresponding
  measured input. Contradictory state/input combinations fail closed.
- The measured spectral response is a square, non-negative, column-stochastic
  operator. It is validated, not silently renormalized.
- Exactly one measurement-noise representation is required. A supplied
  model-discrepancy covariance is checked for symmetry and positive
  semidefiniteness and is added once. The total covariance must be positive
  definite; no numerical jitter is silently inserted.
- Profile inversion requires explicit positive support bounds, positive
  regularization strength for every active profile, and a stated regularized
  quantity/scope. A regularization penalty is not measurement information.
- Local identifiability is diagnosed from a column-scaled Jacobian SVD and
  reports null directions and zero-sensitivity parameters. This is a local
  diagnostic, not proof of global identifiability.

## Integration boundary

These primitives are intended to replace duplicated boundary validation in
Optical Thomson and CTS APIs. Existing APIs retain their own checks until each
call site is migrated with regression tests. Therefore this module alone does
not close TS-07, TS-11, TS-12, TS-14, TS-15, TS-17, or TS-18 at dashboard level,
and does not close physical-model work TS-01 through TS-10.


## Joint spatial-spectral conditioning gate

The joint spatial-spectral optical inverse evaluates identifiability using the
column-scaled Jacobian of measured data only. A numerically full-rank Jacobian
is not sufficient when its condition number exceeds the explicit
`maximum_data_condition_number` input (default `1e8`). In that case the fit
is returned as `ill_conditioned_from_measured_data`; optimizer candidates may
remain available for diagnosis, but primary physical profiles and data-only
parameter covariance are withheld. The threshold is a configurable numerical
stability criterion, not a universal plasma-physics validity boundary.
