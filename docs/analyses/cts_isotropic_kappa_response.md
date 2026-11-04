# Isotropic Kappa electrostatic CTS response

The public advanced-physics dispatcher exposes this kernel as `kappa_electrostatic_susceptibility` via `run_advanced_thomson_physics`; the direct callable remains `tokamak_analysis.cts_kappa.kappa_electron_susceptibility`. The kernel evaluates collisionless longitudinal susceptibility from the exact 1-D marginal of an isotropic 3-D Kappa distribution. It requires kappa > 3/2 and uses a thermal-speed convention with second moment k_B T / m.

The velocity grid and retarded-pole displacement are explicit numerical inputs. The finite-support truncation is reported by the result.

This kernel is not a magnetized dielectric tensor, conserving collisional response, fluctuation-dissipation noise model, relativistic Kappa response, Full-EM CTS three-wave vertex, or unrestricted 3-D EVDF inversion. It does not close TS-10.


The public dispatch route exposes only the susceptibility kernel. It is not a complete CTS spectrum or inverse model, and it must not be interpreted as supplying a Kappa fluctuation-noise/FDT closure.
