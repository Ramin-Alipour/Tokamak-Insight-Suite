import numpy as np


def analyze_wall_dust_population(particle_diameter_um, *, counts=None, material_density_kg_m3: float, histogram_bins=10) -> dict:
    """Estimate dust size distribution and spherical-particle mass."""
    d = np.asarray(particle_diameter_um, dtype=float).ravel()
    if d.size == 0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError('Diameters must be positive and finite.')
    if not np.isfinite(material_density_kg_m3) or material_density_kg_m3 <= 0:
        raise ValueError('Material density must be positive and finite.')
    w = np.ones(d.size) if counts is None else np.asarray(counts, dtype=float).ravel()
    if w.size != d.size or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError('Counts must be nonnegative, finite and match diameters.')
    if isinstance(histogram_bins, bool) or not isinstance(histogram_bins, int) or histogram_bins < 1:
        raise ValueError('histogram_bins must be a positive integer.')
    mass_each = material_density_kg_m3*(np.pi/6)*(d*1e-6)**3
    total_count = float(np.sum(w))
    hist, edges = np.histogram(d, bins=histogram_bins, weights=w)
    return {'diagnostic': 'Wall impurity/dust monitoring', 'particle_diameter_um': d, 'particle_counts': w,
        'particle_mass_kg_each': mass_each, 'total_particle_count': total_count,
        'total_dust_mass_kg': float(np.sum(w*mass_each)),
        'number_weighted_mean_diameter_um': float(np.average(d, weights=w)) if total_count > 0 else None,
        'maximum_diameter_um': float(np.max(d)), 'histogram_counts': hist, 'histogram_edges_um': edges,
        'assumptions_and_limits': ['Spherical particles and uniform material density are assumed.',
            'Measured diameters/counts and image detection limits determine the inferred population.',
            'Composition-dependent density and non-spherical morphology are not modeled.']}
