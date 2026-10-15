from __future__ import annotations
import numpy as np


def analyze_surface_erosion_deposition(initial_surface_m, final_surface_m, surface_area_element_m2, *, material_density_kg_m3: float) -> dict:
    """Quantify surface recession/deposition from registered pre/post surface maps."""
    before = np.asarray(initial_surface_m, dtype=float)
    after = np.asarray(final_surface_m, dtype=float)
    area = np.asarray(surface_area_element_m2, dtype=float)
    if before.size == 0 or not all(np.all(np.isfinite(a)) for a in (before, after, area)):
        raise ValueError('Surface maps and area elements must be finite and nonempty.')
    try:
        before, after, area = np.broadcast_arrays(before, after, area)
    except ValueError as exc:
        raise ValueError('Surface maps and area elements must be broadcast-compatible.') from exc
    if np.any(area < 0) or not np.isfinite(material_density_kg_m3) or material_density_kg_m3 <= 0:
        raise ValueError('Area elements must be nonnegative and material density positive and finite.')
    recession = before-after
    eroded = np.clip(recession, 0, None)
    deposited = np.clip(-recession, 0, None)
    erosion_volume = float(np.sum(eroded*area))
    deposition_volume = float(np.sum(deposited*area))
    return {'diagnostic': 'Surface erosion/deposition', 'signed_recession_m': recession,
        'erosion_depth_m': eroded, 'deposition_thickness_m': deposited,
        'eroded_volume_m3': erosion_volume, 'deposited_volume_m3': deposition_volume,
        'net_material_mass_change_kg': float(material_density_kg_m3*(deposition_volume-erosion_volume)),
        'material_density_kg_m3': float(material_density_kg_m3),
        'assumptions_and_limits': ['Pre/post maps must be spatially registered and share a surface-coordinate convention.',
            'Area elements must account for actual surface geometry and projection.',
            'Mass estimate assumes uniform material density and excludes redeposition outside the mapped area.']}
