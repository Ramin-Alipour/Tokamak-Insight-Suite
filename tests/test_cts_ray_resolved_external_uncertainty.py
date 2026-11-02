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



def test_instrument_response_elements_are_supported_in_joint_external_covariance():
    order = ("instrument_response_0_0", "magnetic_field_T_x",
             "ion_mass_amu_0", "ray_0_1_scattered_direction_y")
    covariance = np.eye(len(order))
    returned_order, returned = validate_external_input_covariance(covariance, order)
    assert returned_order == order
    np.testing.assert_allclose(returned, covariance)


def test_instrument_response_covariance_indices_must_be_nonnegative_integers():
    for name in ("instrument_response_-1_0", "instrument_response_0_x",
                 "instrument_response_0_0_extra"):
        with pytest.raises(ValueError, match="unsupported"):
            validate_external_input_covariance(np.eye(1), (name,))
