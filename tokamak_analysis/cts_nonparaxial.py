"""Exact non-paraxial vector angular-spectrum propagation.

This module provides a Maxwell-transverse vector propagator for a homogeneous,
isotropic, lossless dielectric. It is an exact angular-spectrum solution (no
paraxial/Fresnel expansion), including evanescent components. It is a building
block, not a magnetized-plasma full-wave solver: anisotropic dielectric
coupling, inhomogeneity, reflection, resonant absorption and mode conversion
must be handled by a separate tensor-Maxwell boundary-value solver.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def propagate_nonparaxial_vector_field(
    x_m, y_m, electric_field_V_m, frequency_Hz, refractive_index,
    propagation_distance_m, *, evanescent="decay",
):
    """Propagate a transverse complex E(x,y) field using exact Maxwell k_z.

    Parameters
    ----------
    x_m, y_m : 1-D arrays
        Uniform, strictly increasing transverse coordinates [m].
    electric_field_V_m : complex array, shape (ny,nx,3)
        Complex vector electric field at z=0. The longitudinal component is
        projected to satisfy k·E=0 for every propagating Fourier component.
    frequency_Hz : positive scalar
        Monochromatic frequency.
    refractive_index : positive scalar
        Real refractive index of a homogeneous isotropic medium.
    propagation_distance_m : non-negative scalar
        Forward propagation distance.
    evanescent : {"decay", "discard"}
        Exponentially decay evanescent components or remove them.

    Returns
    -------
    dict
        Propagated vector field, spectral field, transverse wave-number grids,
        and numerical/physical metadata.

    Notes
    -----
    Uses exp(-i omega t + i k·r). For propagating components,
    kz=sqrt((n omega/c)^2-kx^2-ky^2). For evanescent components the branch is
    selected with Im(kz)>=0 so exp(i kz z) decays for z>=0.
    """
    c = 299792458.0
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    field = np.asarray(electric_field_V_m, dtype=complex)
    f = float(frequency_Hz)
    n = float(refractive_index)
    z = float(propagation_distance_m)
    if x.ndim != 1 or y.ndim != 1 or x.size < 4 or y.size < 4:
        raise ValueError("x_m and y_m must be 1-D grids with at least four points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinate grids must be finite")
    dx = np.diff(x); dy = np.diff(y)
    if np.any(dx <= 0) or np.any(dy <= 0) or not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("coordinate grids must be uniformly spaced and strictly increasing")
    if field.shape != (y.size, x.size, 3) or not np.all(np.isfinite(field)):
        raise ValueError("electric_field_V_m must be finite with shape (len(y),len(x),3)")
    if not np.isfinite(f) or f <= 0 or not np.isfinite(n) or n <= 0:
        raise ValueError("frequency_Hz and refractive_index must be positive and finite")
    if not np.isfinite(z) or z < 0:
        raise ValueError("propagation_distance_m must be finite and non-negative")
    if evanescent not in {"decay", "discard"}:
        raise ValueError("evanescent must be 'decay' or 'discard'")

    omega = 2.0 * np.pi * f
    k = n * omega / c
    kx = 2.0 * np.pi * fftfreq(x.size, d=float(dx[0]))
    ky = 2.0 * np.pi * fftfreq(y.size, d=float(dy[0]))
    KX, KY = np.meshgrid(kx, ky)
    kt2 = KX * KX + KY * KY
    kz2 = k * k - kt2
    propagating = kz2 >= 0.0
    KZ = np.sqrt(kz2.astype(complex))
    # Select the outgoing/decaying square-root branch.
    KZ = np.where(np.imag(KZ) < 0.0, -KZ, KZ)

    spectrum = np.stack([fft2(field[..., component]) for component in range(3)], axis=-1)
    # Project each Fourier component onto the plane perpendicular to k.
    # For evanescent waves the complex wavevector is used consistently.
    kvec = np.stack([
        np.broadcast_to(KX, KZ.shape),
        np.broadcast_to(KY, KZ.shape),
        KZ,
    ], axis=-1)
    kdotE = np.sum(kvec * spectrum, axis=-1)
    knorm2 = np.sum(kvec * kvec, axis=-1)
    valid = np.abs(knorm2) > np.finfo(float).tiny
    projected = spectrum.copy()
    projected[valid] -= (
        kvec[valid] * (kdotE[valid] / knorm2[valid])[:, None]
    )
    projected[~valid] = 0.0

    transfer = np.exp(1j * KZ * z)
    if evanescent == "discard":
        transfer = np.where(propagating, transfer, 0.0)
    propagated_spectrum = projected * transfer[..., None]
    propagated = np.stack(
        [ifft2(propagated_spectrum[..., component]) for component in range(3)],
        axis=-1,
    )
    divergence_spectrum = np.sum(kvec * propagated_spectrum, axis=-1)
    scale = max(float(np.linalg.norm(propagated_spectrum)), np.finfo(float).tiny)
    transversality_error = float(np.linalg.norm(divergence_spectrum) / scale)
    return {
        "x_m": x.copy(),
        "y_m": y.copy(),
        "electric_field_V_m": propagated,
        "angular_spectrum_V_m": propagated_spectrum,
        "kx_rad_m": kx,
        "ky_rad_m": ky,
        "kz_rad_m": KZ,
        "propagating_component_mask": propagating,
        "evanescent_component_count": int(np.count_nonzero(~propagating)),
        "transversality_relative_residual": transversality_error,
        "propagation_model": "exact_nonparaxial_vector_angular_spectrum_homogeneous_isotropic",
        "paraxial_approximation_used": False,
        "assumptions": [
            "Homogeneous, isotropic, lossless dielectric with real scalar refractive index.",
            "Monochromatic forward propagation in a Cartesian planar geometry.",
            "Periodic transverse FFT window; edge wrap-around must be controlled by the caller.",
        ],
        "limitations": [
            "Not a magnetized-plasma tensor-dielectric solver.",
            "Does not model spatially varying equilibrium, reflection, cutoffs or resonant layers.",
            "Does not solve O-X, X-B or other plasma mode-conversion layers.",
            "No calibration, equilibrium reconstruction or experimental validation is performed.",
        ],
    }
