from tis.surface_change import analyze_surface_erosion_deposition


def test_surface_erosion_and_deposition_volumes():
    r = analyze_surface_erosion_deposition([1, 1, 1], [0.9, 1.1, 1], [1, 1, 1], material_density_kg_m3=1000)
    assert abs(r['eroded_volume_m3'] - 0.1) < 1e-12
    assert r['deposited_volume_m3'] == 0.1
    assert r['net_material_mass_change_kg'] == 0
