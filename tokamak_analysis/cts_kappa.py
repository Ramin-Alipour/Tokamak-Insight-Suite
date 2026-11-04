import numpy as np

def kappa_pdf(v, kappa, theta):
    return (1.0 + np.asarray(v, dtype=float)**2/(kappa*theta**2))**(-kappa)
