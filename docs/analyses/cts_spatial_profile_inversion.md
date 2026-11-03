# Spatially varying Full-EM CTS profile inversion

The public callable is `tokamak_analysis.cts_spatial_inference.analyze_full_em_cts_spatial_profiles`.

It fits positive, node-wise relative electron-density and electron-temperature
profiles, with optional ion-temperature profiles, against calibrated measured
CTS spectra. Each residual evaluation calls the local magnetized Full-EM
kinetic forward model over the supplied frequency-resolved scattering-volume
quadrature. The model does not collapse a nonuniform plasma into one volume-
averaged Maxwellian.

## Required inputs

- A strictly increasing measured frequency-shift axis and calibrated measured
  spectral-power vector.
- Frequency-resolved ray geometry with a stable `sample_id` at every spatial
  quadrature point, non-negative sample weights summing to one, and a positive
  beam-overlap factor per frequency bin.
- Explicit reference Full-EM plasma/species/field/polarization inputs.
- Exactly one of a full positive-definite spectral covariance or pointwise
  standard uncertainties.
- A measured instrument-response matrix when convolution is required.

## Inversion and uncertainty

The fitted quantities are log-relative node profiles. The default prior is a
second-difference penalty; a researcher-supplied node-space operator and
parameter-specific strengths are accepted. The penalty is reported separately
from measurement covariance. A full spectral covariance is Cholesky-whitened.
The local profile covariance is returned only when the augmented Jacobian has
full column rank; otherwise the fit is explicitly marked locally
non-identifiable and no finite covariance is asserted. An optional strictly
increasing grid of regularization strengths reruns the full inverse and
reports profile spread as regularization-choice sensitivity, separate from
measurement uncertainty.

The response matrix is applied once after the spatially integrated spectrum.
An optional fixed background is added in detector space. The function does
not estimate a background model.

## Scope

This closes the fixed-relative-profile limitation of the ordinary Full-EM
scattering-volume forward path by providing an actual profile inverse. Geometry,
beam weights, mode polarizations and reference bulk parameters remain
conditioned on supplied values. It does not implement a conserving Landau
collision operator, non-paraxial full-vector propagation, or experimental
validation. Synthetic tests establish software behavior only.
