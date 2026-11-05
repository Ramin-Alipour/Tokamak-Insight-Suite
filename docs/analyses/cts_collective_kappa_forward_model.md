# Collective electrostatic CTS with isotropic Kappa electrons

The electron marginal is the exact projection of an isotropic three-dimensional Kappa distribution with second-moment temperature convention:

\[\theta_e^2=\frac{2\kappa-3}{\kappa}\frac{eT_e}{m_e},\quad \kappa>3/2.\]

\[f_{e,1D}(v)=\frac{\Gamma(\kappa)}{\sqrt{\pi\kappa}\theta_e\Gamma(\kappa-1/2)}\left(1+\frac{v^2}{\kappa\theta_e^2}\right)^{-\kappa}.\]

The longitudinal response is obtained from the retarded Landau integral \(\chi_s=-\omega_{ps}^2 k^{-2}\int f_s'(v)/(v-\omega/k-i0^+)\,dv\), with \(\epsilon_L=1+\sum_s\chi_s\). Electron and ion fluctuation terms are screened by \(|\epsilon_L|^{-2}\), giving the collective dynamic structure factor implemented by `collective_kappa_spectral_density`.

The public dispatcher name is `cts_collective_kappa_spectrum`; it is exposed in the advanced Analysis-1 dashboard. This is collisionless, unmagnetized, non-relativistic electrostatic CTS with Maxwellian ions. It is not a magnetized/collisional/Full-EM closure or an inverse Kappa fit. Geometry, ion species, fractions, temperatures and flow vectors remain explicit inputs. Numerical convergence and independent kinetic benchmarking are required before quantitative experimental release.


## Phase-velocity derivative of the Kappa dielectric response

The Kappa Landau kernel also evaluates the derivative of the complex
susceptibility with respect to real phase velocity at fixed positive retarded
pole displacement. From

\[
\chi_e(v_\phi)=-\frac{\omega_{pe}^2}{k^2}
\int_{-\infty}^{\infty}\frac{f'_e(v)}{v-v_\phi-i\eta}\,dv,
\qquad \eta>0,
\]

differentiation gives

\[
\frac{\partial\chi_e}{\partial v_\phi}
=-\frac{\omega_{pe}^2}{k^2}
\int_{-\infty}^{\infty}\frac{f'_e(v)}
{(v-v_\phi-i\eta)^2}\,dv.
\]

Both the original response and its derivative integrate the exact analytic
Kappa derivative over the complete velocity axis. The resonance neighbourhood
is transformed to a dimensionless coordinate before adaptive quadrature; the
derivative's additional inverse-velocity factor is retained explicitly. The
API returns the complex derivative and separate real/imaginary quadrature
error estimates. This derivative supports dielectric root sensitivity and
local identifiability analysis, but does not add collisions, magnetization,
relativistic kinetics or a Full-EM CTS vertex.


At fixed scattering wavenumber, the angular-frequency sensitivity follows
from the chain rule \(v_\phi=\omega/k\):
\[
\left.\frac{\partial\chi_e}{\partial\omega}\right|_k
=\frac{1}{k}\frac{\partial\chi_e}{\partial v_\phi}.
\]
The solver returns this quantity as
`susceptibility_angular_frequency_derivative_s`; it is not a derivative at
fixed phase velocity or a full multi-species dielectric derivative.
