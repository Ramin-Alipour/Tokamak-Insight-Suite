from __future__ import annotations
import numpy as np


def analyze_particle_transport(radius_m, density_m3, particle_flux_m2_s, *, gradient_floor: float = 0.0) -> dict:
    """Estimate local effective particle diffusivity D=-Gamma/(dn/dr)."""
    r = np.asarray(radius_m, dtype=float).ravel()
    n = np.asarray(density_m3, dtype=float).ravel()
    flux = np.asarray(particle_flux_m2_s, dtype=float).ravel()
    if r.size < 3 or len({r.size, n.size, flux.size}) != 1 or not all(np.all(np.isfinite(a)) for a in (r, n, flux)):
        raise ValueError('Radius, density and flux must be finite equal-length profiles with at least three points.')
    if np.any(np.diff(r) <= 0) or np.any(n < 0):
        raise ValueError('Radius must increase strictly and density must be nonnegative.')
    if not np.isfinite(gradient_floor) or gradient_floor < 0:
        raise ValueError('gradient_floor must be finite and nonnegative.')
    grad = np.gradient(n, r, edge_order=2)
    valid = np.abs(grad) > gradient_floor
    diffusivity = np.full_like(grad, np.nan)
    diffusivity[valid] = -flux[valid]/grad[valid]
    return {'diagnostic': 'Particle flux / effective diffusion', 'radius_m': r,
        'density_gradient_m-4': grad, 'particle_flux_m-2_s-1': flux,
        'effective_diffusivity_m2_s-1': diffusivity, 'valid_diffusivity_mask': valid,
        'assumptions_and_limits': ['Local Fickian estimate D=-Gamma/(dn/dr); convective pinch and nonlocal transport are not separated.',
            'Flux and gradient must use a consistent radial coordinate and sign convention.',
            'Values near flat gradients are ill-conditioned; negative D can indicate pinch, sign mismatch or model inadequacy.']}
