# Collective electrostatic CTS with isotropic Kappa electrons

The electron marginal is the exact projection of an isotropic three-dimensional Kappa distribution with second-moment temperature convention:

\[\theta_e^2=\frac{2\kappa-3}{\kappa}\frac{eT_e}{m_e},\quad \kappa>3/2.\]

\[f_{e,1D}(v)=\frac{\Gamma(\kappa)}{\sqrt{\pi\kappa}\theta_e\Gamma(\kappa-1/2)}\left(1+\frac{v^2}{\kappa\theta_e^2}\right)^{-\kappa}.\]

The longitudinal response is obtained from the retarded Landau integral \(\chi_s=-\omega_{ps}^2 k^{-2}\int f_s'(v)/(v-\omega/k-i0^+)\,dv\), with \(\epsilon_L=1+\sum_s\chi_s\). Electron and ion fluctuation terms are screened by \(|\epsilon_L|^{-2}\), giving the collective dynamic structure factor implemented by `collective_kappa_spectral_density`.

The public dispatcher name is `cts_collective_kappa_spectrum`; it is exposed in the advanced Analysis-1 dashboard. This is collisionless, unmagnetized, non-relativistic electrostatic CTS with Maxwellian ions. It is not a magnetized/collisional/Full-EM closure or an inverse Kappa fit. Geometry, ion species, fractions, temperatures and flow vectors remain explicit inputs. Numerical convergence and independent kinetic benchmarking are required before quantitative experimental release.
