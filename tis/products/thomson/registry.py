"""Product-owned executable map for Thomson Scattering analyses.

The map is deliberately local to the specialist product: resolving a Thomson
solver must not import a callable from the legacy tokamak_analysis registry.
Scientific contracts remain published in the canonical diagnostic catalogue;
this module owns only Python execution routing.
"""
from __future__ import annotations

THOMSON_ENTRY_POINTS = {
    1: ("thomson_workflow", "analyze_thomson_spectra"),
    30: ("cts_analysis", "analyze_cts_spectrum_model"),
    46: ("thomson_workflow", "reconstruct_thomson_profile_from_spectra"),
    81: ("thomson_distribution", "analyze_thomson_spectral_distribution"),
    82: ("thomson_workflow", "fit_gyrotropic_thomson_spectra"),
    83: ("thomson_workflow", "fit_general_temperature_tensor_thomson_spectra"),
    84: ("cts_full_physics", "analyze_landau_fdt_full_em_spectrum_converged"),
    85: ("advanced_inference", "reconstruct_relativistic_optical_thomson_3d_evdf"),
    86: ("thomson_temperature_tensor", "compare_general_temperature_tensor_thomson_models"),
    87: ("thomson_scattering", "infer_thomson_path_averaged_density"),
}

THOMSON_ANALYSIS_IDS = frozenset(THOMSON_ENTRY_POINTS)
