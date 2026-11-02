import numpy as np
import pytest

from tokamak_analysis.cts_external_uncertainty import validate_external_input_covariance


def test_ray_resolved_cts_external_parameter_names_are_supported():
    order = (
        "ray_0_0_magnetic_field_T_x",
        "ray_0_0_incident_direction_z",
        "ray_0_1_scattered_direction_y",
        "ray_1_beam_overlap_relative",
        "ion_mass_amu_0",
    )
    covariance = np.eye(len(order))
    returned_order, returned = validate_external_input_covariance(covariance, order)
    assert returned_order == order
    np.testing.assert_allclose(returned, covariance)


def test_ray_resolved_cts_external_parameter_names_reject_bad_components():
    for name in ("ray_0_magnetic_field_T_x", "ray_0_0_electric_field_T_x",
                 "ray_0_0_incident_direction_q", "ray_0_beam_overlap_relative_extra"):
        with pytest.raises(ValueError, match="unsupported"):
            validate_external_input_covariance(np.eye(1), (name,))
