"""Public execution API for the Thomson Scattering specialist product.

Each function routes to the canonical registered scientific workflow. The
registry remains the source of truth for callable ownership and contracts.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "thomson-scattering"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a registered Thomson Scattering workflow."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def _run(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    return run_analysis(analysis_id, *args, **kwargs)


def analyze_spectra(*args: Any, **kwargs: Any) -> Any:
    """Run the registered optical Thomson spectrum-to-physics pipeline (analysis 1).

    The canonical workflow returns model-specific plasma inference together
    with a projected q-parallel EVDF when its inverse is enabled. Dimensional
    third/fourth central velocity moments and their covariance are exposed as
    projected-marginal quantities; a single view does not identify the full
    three-dimensional heat flux or unrestricted EVDF.
    """
    return _run(1, *args, **kwargs)


def analyze_cts(*args: Any, **kwargs: Any) -> Any:
    """Run collective/coherent Thomson scattering inference (entry 30)."""
    return _run(30, *args, **kwargs)


def reconstruct_profiles(*args: Any, **kwargs: Any) -> Any:
    """Reconstruct spatial profiles from measured Thomson spectra (entry 46)."""
    return _run(46, *args, **kwargs)


def analyze_spectral_distribution(*args: Any, **kwargs: Any) -> Any:
    """Analyze measured Thomson spectral distribution (entry 81)."""
    return _run(81, *args, **kwargs)


def fit_gyrotropic(*args: Any, **kwargs: Any) -> Any:
    """Fit multi-view gyrotropic anisotropic Thomson spectra (entry 82)."""
    return _run(82, *args, **kwargs)


def fit_temperature_tensor(*args: Any, **kwargs: Any) -> Any:
    """Fit a general non-gyrotropic 3-D electron temperature tensor (entry 83)."""
    return _run(83, *args, **kwargs)


def analyze_landau_fdt_convergence(*args: Any, **kwargs: Any) -> Any:
    """Run the registered Landau-FDT Full-EM convergence solver (entry 84)."""
    return _run(84, *args, **kwargs)


def reconstruct_relativistic_evdf(*args: Any, **kwargs: Any) -> Any:
    """Reconstruct relativistic 3-D EVDF from multi-view optical spectra (entry 85)."""
    return _run(85, *args, **kwargs)


def compare_temperature_tensor_models(*args: Any, **kwargs: Any) -> Any:
    """Compare Maxwellian/Kappa tensor model families (entry 86)."""
    return _run(86, *args, **kwargs)


def infer_path_averaged_density(*args: Any, **kwargs: Any) -> Any:
    """Infer path-averaged density from correlated optical calibration (entry 87)."""
    return _run(87, *args, **kwargs)

def fit_two_population_relativistic_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Fit the explicitly parameterized two-population relativistic TS model."""
    from .physics.thomson_workflow import fit_two_population_relativistic_thomson_spectrum
    return fit_two_population_relativistic_thomson_spectrum(*args, **kwargs)


def fit_n_population_relativistic_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Fit the explicitly parameterized finite N-population relativistic TS model."""
    from .physics.thomson_workflow import fit_n_population_relativistic_thomson_spectrum
    return fit_n_population_relativistic_thomson_spectrum(*args, **kwargs)


def fit_kinetic_spectrum_with_background(*args: Any, **kwargs: Any) -> Any:
    """Jointly infer a non-collective kinetic Thomson spectrum and structured background."""
    from .physics.thomson_workflow import fit_noncollective_kinetic_thomson_with_background
    return fit_noncollective_kinetic_thomson_with_background(*args, **kwargs)


def propagate_full_em_cts_ray_geometry_uncertainty(*args: Any, **kwargs: Any) -> Any:
    """Propagate supplied ray/equilibrium geometry uncertainty through Full-EM CTS refits."""
    from .physics.thomson_workflow import propagate_full_em_cts_ray_geometry_uncertainty as propagate
    return propagate(*args, **kwargs)

__all__ = [
    "PRODUCT_ID",
    "run_analysis",
    "analyze_spectra",
    "analyze_cts",
    "reconstruct_profiles",
    "analyze_spectral_distribution",
    "fit_gyrotropic",
    "fit_temperature_tensor",
    "analyze_landau_fdt_convergence",
    "reconstruct_relativistic_evdf",
    "compare_temperature_tensor_models",
    "infer_path_averaged_density",
    "fit_two_population_relativistic_spectrum",
    "fit_n_population_relativistic_spectrum",
    "fit_kinetic_spectrum_with_background",
    "propagate_full_em_cts_ray_geometry_uncertainty",
]
