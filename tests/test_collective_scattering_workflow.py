import numpy as np
from tis.collective_scattering import fit_collective_scattering_spectrum


def test_template_fit():
    x = np.linspace(-5, 5, 51)
    a = np.exp(-0.5*x*x)
    b = np.exp(-0.5*(x-2)**2)
    matrix = np.column_stack([a, b])
    y = matrix @ np.array([3.0, 1.5])
    result = fit_collective_scattering_spectrum(x, y, matrix, ['a', 'b'], uncertainty=np.ones_like(y)*0.01)
    np.testing.assert_allclose(result['nonnegative_amplitudes'], [3.0, 1.5], rtol=1e-5)
