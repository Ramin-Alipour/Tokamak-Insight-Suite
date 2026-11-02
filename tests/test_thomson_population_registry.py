from tokamak_analysis.analysis_registry import ANALYSES


def test_two_population_thomson_subanalysis_is_registered():
    analysis = next(item for item in ANALYSES if item["id"] == 1)
    modes = analysis["execution_contract"]["workflow_modes"]
    population = next(item for item in modes if item["name"] == "two_population_relativistic_optical_thomson")
    assert population["function"] == "tokamak_analysis.thomson_population_mixture.fit_two_population_relativistic_thomson_spectrum"
    assert "hot_population_fraction" in population["outputs"]
    assert "absolute density" in population["scope"]
