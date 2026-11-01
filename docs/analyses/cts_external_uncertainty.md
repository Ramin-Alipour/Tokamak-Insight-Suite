# Full-EM CTS external-input uncertainty contract

The local external-input covariance utility provides a strict ordered parameter contract for researcher-supplied uncertainty in magnetic-field components, incident/scattered direction components, and real/imaginary components of incident/scattered complex polarization vectors. It validates symmetric positive-semidefinite covariance matrices and generates reproducible joint perturbation draws.

This utility is not yet connected to the Full-EM inverse fit. It does not claim that external-input uncertainty has been propagated into fitted plasma parameters. Frequency-resolved ray/equilibrium covariance, composition covariance, and cross-covariance with instrument response remain outside this utility. No instrument calibration or empirical validation is performed.
