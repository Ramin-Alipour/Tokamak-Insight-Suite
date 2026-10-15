import numpy as np
from .core import regularized_tomography


def analyze_neutron_tomography(measured_projections, response_matrix, image_shape, *, pixel_area_m2: float | None = None, regularization=1e-2, uncertainty=None) -> dict:
    """Reconstruct a nonnegative neutron-emission image from a supplied response matrix."""
    y = np.asarray(measured_projections, dtype=float).ravel()
    A = np.asarray(response_matrix, dtype=float)
    if len(image_shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in image_shape):
        raise ValueError('image_shape must contain two integers of at least two pixels.')
    if A.ndim != 2 or A.shape != (y.size, image_shape[0]*image_shape[1]) or y.size < 2:
        raise ValueError('Response matrix dimensions must match projection data and image_shape.')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(A)) or np.any(y < 0) or np.any(A < 0):
        raise ValueError('Projection data and response matrix must be finite and nonnegative.')
    if pixel_area_m2 is not None and (not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0):
        raise ValueError('pixel_area_m2 must be positive and finite.')
    sigma = None if uncertainty is None else np.asarray(uncertainty, dtype=float).ravel()
    if sigma is not None and (sigma.size != y.size or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
        raise ValueError('uncertainty must be positive, finite and match projections.')
    fit = regularized_tomography(A, y, lam=regularization, nonnegative=True, measurement_sigma=sigma, regularization_order=2)
    image = np.asarray(fit['solution'], dtype=float).reshape(image_shape)
    total = float(np.sum(image)*pixel_area_m2) if pixel_area_m2 is not None else None
    return {'diagnostic': 'Neutron imaging / tomography', 'reconstructed_emission_image': image,
        'integrated_emission_proxy': total, 'pixel_area_m2': pixel_area_m2,
        'residual_norm': fit['residual_norm'], 'regularization': float(regularization),
        'assumptions_and_limits': ['Response matrix must represent detector geometry and neutron transport/attenuation.',
            'Image reconstruction is ill-posed; resolution and artifacts depend on view coverage and regularization.',
            'Absolute source strength requires calibrated detector efficiency and attenuation model.']}
