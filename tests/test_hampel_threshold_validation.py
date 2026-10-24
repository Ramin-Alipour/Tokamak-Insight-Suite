import numpy as np
import pytest

from tis.signals import hampel_despike


@pytest.mark.parametrize("n_sigmas", [0, -1, np.nan, np.inf, True, "6"])
def test_hampel_rejects_invalid_threshold(n_sigmas):
    with pytest.raises(ValueError, match="n_sigmas"):
        hampel_despike(np.arange(15.0), n_sigmas=n_sigmas)
