# Thomson Scattering closure execution note

This file records the implementation boundary for the remaining Thomson closure work. It is not evidence that the listed code changes have been implemented.

The remaining items require coordinated changes to forward models, inversion APIs, dashboard registration, and tests. In particular, a one-dimensional projected relativistic Maxwell-Juttner marginal must not be described as an arbitrary three-dimensional EVDF reconstruction. External-input covariance must remain distinct from measurement-noise covariance. PSF/mixing belongs in the forward operator, and regularization must not be treated as additional data identifiability.
