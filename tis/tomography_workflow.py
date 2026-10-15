from __future__ import annotations
from typing import Any
import numpy as np
from .core import build_tomography_matrix_2d, regularized_tomography


def bolometric_tomography_workflow(ray_start_xz: Any, ray_end_xz: Any, x_edges_m: Any, z_edges_m: Any, measurements: Any, *, regularization: float = 1e-2, nonnegative: bool = True, measurement_sigma: Any | None = None, regularization_order: int = 1) -> dict[str, Any]:
    """Build ray-length geometry and solve a regularized tomography inverse problem."""
    matrix = build_tomography_matrix_2d(ray_start_xz, ray_end_xz, x_edges_m, z_edges_m)
    values = np.asarray(measurements, dtype=float).ravel()
    if len(values) != matrix.shape[0] or not np.all(np.isfinite(values)):
        raise ValueError('Supply one finite line-integrated value per ray.')
    result = regularized_tomography(matrix, values, lam=regularization, nonnegative=nonnegative, measurement_sigma=measurement_sigma, regularization_order=regularization_order)
    x_edges = np.asarray(x_edges_m, dtype=float).ravel()
    z_edges = np.asarray(z_edges_m, dtype=float).ravel()
    nx, nz = len(x_edges) - 1, len(z_edges) - 1
    return {**result, 'response_matrix_path_lengths_m': matrix, 'emissivity_proxy_grid': np.asarray(result['solution']).reshape(nz, nx), 'x_edges_m': x_edges, 'z_edges_m': z_edges, 'flattening_order': 'iz * nx + ix', 'interpretation': 'Conditional on supplied geometry, calibration, units and regularization; no detector calibration or unique solution is inferred.'}
