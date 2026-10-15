from tis.dust_monitor import analyze_wall_dust_population


def test_dust_population_mass_and_histogram():
    r = analyze_wall_dust_population([10, 20, 30], counts=[2, 1, 1], material_density_kg_m3=2000, histogram_bins=3)
    assert r['total_particle_count'] == 4
    assert r['total_dust_mass_kg'] > 0
    assert sum(r['histogram_counts']) == 4
