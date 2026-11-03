from tokamak_analysis.analysis_registry import ANALYSES


def test_two_population_thomson_subanalysis_is_registered():
    analysis = next(item for item in ANALYSES if item["id"] == 1)
    modes = analysis["execution_contract"]["workflow_modes"]
    population = next(item for item in modes if item["name"] == "two_population_relativistic_optical_thomson")
    assert population["function"] == "tokamak_analysis.thomson_population_mixture.fit_two_population_relativistic_thomson_spectrum"
    assert "hot_population_fraction" in population["outputs"]
    assert "absolute density" in population["scope"]


def test_two_population_registry_marks_optimizer_seeds_optional_and_all_or_none():
    analysis = next(item for item in ANALYSES if item["id"] == 1)
    modes = analysis["execution_contract"]["workflow_modes"]
    population = next(item for item in modes if item["name"] == "two_population_relativistic_optical_thomson")
    inputs = " ".join(population["inputs"]).lower()
    assert "optional all-or-none optimizer seeds" in inputs
    assert "automatic deterministic multistart" in inputs
    assert "never reported as inferred measurements" in population["scope"]


def test_n_population_thomson_inverse_is_registered_in_canonical_analysis_one():
    analysis = next(item for item in ANALYSES if item["id"] == 1)
    modes = analysis["execution_contract"]["workflow_modes"]
    population = next(
        item for item in modes
        if item["name"] == "n_population_relativistic_optical_thomson"
    )
    assert population["function"] == "tokamak_analysis.thomson_workflow.fit_n_population_relativistic_thomson_spectrum"
    assert "population_count" in " ".join(population["inputs"])
    assert "joint_parameter_covariance" in " ".join(population["outputs"])
    assert "not an unrestricted" in population["scope"]
