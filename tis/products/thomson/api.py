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
    with a projected q-parallel EVDF when its inverse is enabled. For the
    non-collective isotropic-Kappa branch, supplied laser-wavelength and
    scattering-angle calibration uncertainties (including their cross-covariance)
    are propagated into the fitted temperature, q-parallel flow and analytic
    projected-Kappa moment closure. Dimensional third/fourth central velocity
    moments and their covariance remain projected-marginal quantities. The
    non-relativistic Maxwellian/Kappa branches additionally report the
    identifiable q-direction pressure component and one-dimensional
    projected kinetic-energy flux when the primary fit independently
    identifies absolute density. Total absolute uncertainty is withheld
    without the joint density/EVDF covariance; a single view does not identify
    the full three-dimensional pressure tensor, heat flux or unrestricted EVDF.
    """
    return _run(1, *args, **kwargs)


def analyze_cts(*args: Any, **kwargs: Any) -> Any:
    """Run collective/coherent Thomson scattering inference (entry 30)."""
    return _run(30, *args, **kwargs)


def reconstruct_profiles(*args: Any, **kwargs: Any) -> Any:
    """Reconstruct spatial profiles from measured Thomson spectra (entry 46)."""
    return _run(46, *args, **kwargs)


def analyze_spectral_distribution(*args: Any, **kwargs: Any) -> Any:
    """Analyze measured Thomson spectral distribution (entry 81).

    Set projected_evdf_options to reconstruct the q-parallel EVDF from the
    same calibrated spectrum. Optional electron_density_m3 and
    electron_density_uncertainty_m3 inside that mapping enable the
    one-dimensional projected parallel kinetic-energy-flux estimate; the
    returned quantity is explicitly not the full 3-D electron heat flux.
    The same options expose P_qq = m_e n_e sigma_q^2, the q-direction
    pressure component, with optional density/EVDF-variance cross-covariance.
    For relativistic kinematics, the same independent density also enables
    absolute laboratory-frame projected kinetic-energy flux; supply
    electron_density_flux_per_density_cross_covariance_W_m2 when its
    covariance with the EVDF flux-per-density estimate is known.
    The physical-moment result also returns exact piecewise-linear EVDF
    moments through sixth order, their full first-order covariance, a separate
    one-edge-cell velocity-grid truncation sensitivity diagnostic (not folded
    into statistical uncertainty), and the cumulative q-parallel random-energy
    distribution with covariance. With an
    independent density and uncertainty, it returns P_qq and q_q^(1D) plus
    their joint covariance, retaining Cov(sigma_q^2, mu_3,q) and only
    caller-supplied density cross-covariances. These are one-dimensional
    q-direction quantities, not the full 3-D pressure tensor or heat flux;
    the non-relativistic closures are withheld for relativistic kinematics.
    """
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
    """Infer Thomson sightline density (entry 87).

    A scalar scattered_power_W returns one kernel-weighted path average.
    A channel vector returns one weighted path average per sightline and can
    propagate the full joint covariance of all channel powers and path-resolved
    optical factors. Neither mode claims a pointwise density profile.
    """
    return _run(87, *args, **kwargs)

def reconstruct_nonrelativistic_3d_evdf(*args: Any, **kwargs: Any) -> Any:
    """Reconstruct a non-negative 3-D EVDF and its identifiable velocity moments (entry 88).

    The registered solver returns the mean-flow vector, full central velocity
    covariance, principal kinetic-temperature tensor, third central-moment
    tensor and (inside the declared classical domain) heat flux per density.
    When the data-only multi-view operator identifies the EVDF, covariance
    propagation includes the normalized-distribution Jacobian; the returned
    temperature-tensor covariance is in row-major [xx,xy,xz,yx,yy,yz,zx,zy,zz]
    component order. Regularization-only candidates are never promoted to
    primary physical measurements.
    """
    return _run(88, *args, **kwargs)


def fit_optical_thomson_poisson_counts(
    *args: Any, distribution_model: str = "maxwellian", **kwargs: Any
) -> Any:
    """Run Analysis-1 finite-bin Poisson inference for Maxwellian or Kappa EVDFs.

    Kappa fits estimate kappa jointly with the spectral centroid, standard
    deviation, amplitude and detector-background coefficients. The Kappa
    branch requires kappa > 3/2 for a finite second-moment temperature.
    """
    if distribution_model not in {"maxwellian", "kappa"}:
        raise ValueError("distribution_model must be 'maxwellian' or 'kappa'")
    kwargs["likelihood"] = "independent_Poisson"
    kwargs["distribution_model"] = distribution_model
    return _run(1, *args, **kwargs)


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


def run_advanced_physics(model: str, *args: Any, **kwargs: Any) -> Any:
    """Execute an explicitly named advanced Thomson physics kernel via the product API.

    This boundary is for advanced, composable physical kernels that are not
    separate catalog analysis IDs. Catalog workflows must continue to use
    run_analysis(), which enforces product ownership and registered contracts.
    """
    from .physics.cts_full_physics import run_advanced_thomson_physics
    return run_advanced_thomson_physics(model, *args, **kwargs)


def propagate_full_em_cts_ray_geometry_uncertainty(*args: Any, **kwargs: Any) -> Any:
    """Propagate supplied ray/equilibrium geometry uncertainty through Full-EM CTS refits."""
    from .physics.thomson_workflow import propagate_full_em_cts_ray_geometry_uncertainty as propagate
    return propagate(*args, **kwargs)

__all__ = [
    "PRODUCT_ID",
    "run_analysis",
    "run_advanced_physics",
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
    "reconstruct_nonrelativistic_3d_evdf",
    "fit_optical_thomson_poisson_counts",
    "fit_two_population_relativistic_spectrum",
    "fit_n_population_relativistic_spectrum",
    "fit_kinetic_spectrum_with_background",
    "propagate_full_em_cts_ray_geometry_uncertainty",
]
