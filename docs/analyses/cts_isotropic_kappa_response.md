# Isotropic Kappa electrostatic CTS response

The tokamak_analysis.cts_kappa.kappa_electron_susceptibility function evaluates collisionless longitudinal susceptibility from the exact 1-D marginal of an isotropic 3-D Kappa distribution. It requires kappa > 3/2 and uses a thermal-speed convention with second moment k_B T / m.

The velocity grid and retarded-pole displacement are explicit numerical inputs. The finite-support truncation is reported by the result.

This kernel is not a magnetized dielectric tensor, conserving collisional response, fluctuation-dissipation noise model, relativistic Kappa response, Full-EM CTS three-wave vertex, or unrestricted 3-D EVDF inversion. It does not close TS-10.
