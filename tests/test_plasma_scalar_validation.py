"""Regression tests for strict numeric validation in plasma scaling APIs."""
import numpy as np
import pytest

from tis.plasma import (
    beta_metrics,
    confinement_summary,
    greenwald,
    ipb98y2_confinement_time,
    q95_iter_formula,
    spitzer_resistivity,
)

_BAD = [True, np.bool_(True), "1.0", np.nan, np.inf, 10**1000]


@pytest.mark.parametrize("bad", _BAD)
@pytest.mark.parametrize(
    "call",
    [
        lambda bad: spitzer_resistivity(bad),
        lambda bad: q95_iter_formula(bad, 2.0, 1.0, 0.3, 1.5),
        lambda bad: beta_metrics(1e19, 100.0, 100.0, bad, 2.0, 0.3),
        lambda bad: greenwald(1e19, bad, 0.3),
        lambda bad: ipb98y2_confinement_time(bad, 2.0, 1.0, 5.0, 2.0, 1.0, 0.3, 1.5),
    ],
)
def test_plasma_scalings_reject_invalid_scalar_inputs(call, bad):
    with pytest.raises(ValueError):
        call(bad)


@pytest.mark.parametrize("field,bad", [
    ("ip_A", True),
    ("bt_T", "2.0"),
    ("p_loss_W", np.inf),
    ("ne_line_avg_m3", 10**1000),
    ("delta", False),
])
def test_confinement_summary_validates_before_arithmetic(field, bad):
    kwargs = dict(
        ip_A=1e6, bt_T=2.0, r_m=1.0, a_m=0.3, kappa=1.5,
        ne_line_avg_m3=5e19, p_loss_W=1e6, mass_amu=2.0,
    )
    kwargs[field] = bad
    with pytest.raises(ValueError):
        confinement_summary(**kwargs)
