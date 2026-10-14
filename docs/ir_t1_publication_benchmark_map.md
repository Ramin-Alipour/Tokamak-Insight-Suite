# IR-T1 publication benchmark map

This file fixes the scientific target of the next validation pass: reproduce the main plotted results from the six IR-T1 papers using the equations and analysis methods stated in those papers, then compare the calculated outputs with independently extracted figure values. The source PDFs remain in the user's Library; the software repository should not ship the original PDFs or full digitized paper datasets.

## Article-by-article comparison targets

| Paper | Main plotted result to benchmark | Analysis needed in TIS | Comparison rule |
|---|---|---|---|
| Alipour et al., *First investigation on plasma impurities of the IR-T1 tokamak*, AIP Advances 7, 115303 (2017), DOI [10.1063/1.4990869](https://doi.org/10.1063/1.4990869) | Calibrated visible spectra, identified line centers/widths, and derived impurity/temperature/velocity values shown in figures | Wavelength calibration; background/response handling; Gaussian/Voigt fit; instrument-width correction; Doppler conversion | Compare the same line, wavelength unit, fit interval and broadening assumptions. Digitization uncertainty must be included when only a figure is available. |
| Alipour et al., *Investigation on the Hard X-ray Radiations of the IR-T1 Tokamak Plasma: Electric and Magnetic Perspectives*, Brazilian Journal of Physics 47, 567–574 (2017), DOI [10.1007/s13538-017-0536-6](https://doi.org/10.1007/s13538-017-0536-6) | X-ray spectra in the published time windows; average runaway-electron energy; temporal relationship to dominant Mirnov components | Energy-axis calibration; spectrum/count summary; SVD of synchronized Mirnov channels; Morlet time-frequency analysis | Keep energy calibration, detector response/efficiency and acquisition windows explicit. A plotted spectrum alone does not recover the original counts. |
| Alipour et al., *Design and fabrication of a new compound probe for plasma flux measurement in IR-T1 tokamak*, RSI 88, 093516 (2017), DOI [10.1063/1.4994037](https://doi.org/10.1063/1.4994037) | Probe I–V/collector response and reported edge values, including (T_e), plasma potential and parallel/perpendicular Mach-number estimates | Probe-specific I–V model; collector geometry and sign conventions; derived flow quantities | Do not compare a generic Langmuir fit to a compound-probe result unless the collection/sheath model and electrical conventions match. Published summary values are context, not raw I–V data. |
| Alipour et al., *Investigation on the effect of pressure on turbulent transports of the IR-T1 Tokamak plasma*, EPJ D 71, 60 (2017), DOI [10.1140/epjd/e2017-70563-6](https://doi.org/10.1140/epjd/e2017-70563-6) | Figure 3 time histories: (E_r), (E_p), radial/poloidal turbulent transport and Reynolds-stress product across 1.9, 2.3 and 2.7 Torr | Potential-gradient calculation; explicit (E\times B) velocity mapping; (n_i v_r), (n_i v_p), and (v_r v_p); separate fluctuation covariance | Compare same pressure, time interval, units, sign convention and magnetic field. Distinguish mean instantaneous products from covariance of fluctuations. |
| Alipour et al., *Effects of the location of a biased limiter on turbulent transport in the IR-T1 tokamak plasma*, EPJ D 71, 228 (2017), DOI [10.1140/epjd/e2017-80215-6](https://doi.org/10.1140/epjd/e2017-80215-6) | Plotted plasma-current, electric-field, transport and Reynolds-stress traces for limiter positions and ±200 V bias | Same edge-transport workflow plus consistent windowing and baseline/control comparisons | Reproduce each position/bias condition separately; percentage changes must use a documented baseline and avoid dividing by near-zero reference values. |
| Alipour & Ghanbari, *Magnetic evaluation of hydrogen pressures changes on MHD fluctuations in IR-T1 tokamak plasma*, EPJ D 72, 75 (2018), DOI [10.1140/epjd/e2018-90033-y](https://doi.org/10.1140/epjd/e2018-90033-y) | Mirnov PSD peaks and relative spectral levels; Morlet/wavelet time-frequency maps; SVD spatial/temporal components and mode-energy fractions for 1.9, 2.5 and 2.9 Torr | Uniform-sampling QC; FFT/Welch PSD; SVD; Morlet CWT; explicit sensor order and time windows | Compare peak frequency and relative power with the plotted figure. SVD signs are arbitrary; mode ordering, sensor order, normalization and window selection must match before comparing component shapes. |

## First figure-derived consistency check: compound probe

Source: Alipour et al., RSI 88, 093516 (2017), DOI [10.1063/1.4994037](https://doi.org/10.1063/1.4994037), Fig. 10(a) (smoothed parallel Mach number), Fig. 11(a) (parallel flow velocity), and Eq. (4).

A provisional pixel-based digitization of the two plotted curves over the full 0–30 ms discharge gives:
- Figure 10(a): mean smoothed (M_{\\parallel}\\approx0.511).
- Figure 11(a): mean (V_{\\parallel}\\approx17.57\\;\\mathrm{km\\,s^{-1}}).

The paper's abstract reports rounded values (~0.5 and ~17 km/s), consistent with these approximate digitizations. The consistency calculation is assumption-sensitive:
- With the simplified electron-temperature-only estimate (T_e=14) eV and (m_i=1.0073) amu, (c_s\\approx36.620\\;\\mathrm{km\\,s^{-1}}); (M c_s\\approx18.713) km/s, 6.50% above the digitized velocity, while (V/c_s\\approx0.480).
- If one additionally assumes (T_i/T_e=0.2), then (c_s\\approx40.115\\;\\mathrm{km\\,s^{-1}}); (M c_s\\approx20.499) km/s, 16.67% above the digitized velocity, while (V/c_s\\approx0.438), 14.29% below the digitized Mach mean.

The probe-paper's exact sound-speed temperature convention and the pointwise inputs used for the plotted velocity must be verified before choosing between these cases; (T_i/T_e=0.2) is shown only as a conditional scenario, not a confirmed input from this paper. Averages digitized independently from plotted traces can also differ from a calculation made from the original paired time series. This is an **unresolved consistency flag**, not proof that the software or paper is wrong. Next steps are to refine both curve digitizations with uncertainty, inspect the equation/definitions in the paper, and compare pointwise values if original data can be obtained. The regression test keeps the alternative assumptions explicit rather than forcing a match.

## Doppler-temperature reconstruction from the published tables

Source: Alipour et al., *AIP Advances* 7, 115303 (2017), DOI [10.1063/1.4990869](https://doi.org/10.1063/1.4990869), Table VI (corrected Doppler FWHM) and Table VII (ion temperature).

The published Table VI widths are Doppler **FWHM**, not Gaussian standard deviations. The reconstruction therefore uses (\\sigma_\\lambda=\\Delta\\lambda_{\\rm FWHM}/[2\\sqrt{2\\ln2}]) before applying (T_i=m_i c^2/e\\,(\\sigma_\\lambda/\\lambda_0)^2). This is equivalent to the paper's Eq. (15)–(16). A regression test reconstructs all 15 Table VII temperatures from the rounded Table VI widths and Table VII masses, and independently checks the SI formula against the software routine. The reported rounding is reproduced within 0.16 eV; examples are Ti I: 3.65 eV calculated vs 3.66 eV published, and O V: 85.63 eV vs 85.75 eV.

This is a successful **table-to-formula reconstruction**, not a fresh fit of the experimental spectra: the paper's underlying CCD samples and the full Gaussian-fit residuals are not available in these table values. It checks the FWHM-to-sigma convention and the temperature conversion, but does not independently confirm line isolation, instrument correction, calibration, or neglected Zeeman/Stark/natural broadening. The new test is `tests/test_ir_t1_doppler_table_reconstruction.py`.

## Figure-derived hard-X-ray result check

Source: Alipour et al., *Brazilian Journal of Physics* 47, 567–574 (2017), DOI [10.1007/s13538-017-0536-6](https://doi.org/10.1007/s13538-017-0536-6), Fig. 6 (0–5 ms spectrum) and Table 1.

The five visible event energies in the 0–5 ms panel were provisionally digitized as 318.68, 620.88, 673.08, 750.00 and 925.82 keV. Treating each plotted event as one count, TIS `xray_spectrum_summary(..., bin_representation="counts_per_bin")` gives a mean of **657.69 keV**. Table 1 reports 5 counts and 3301 keV total energy, corresponding to **660.20 keV**; the difference is **−0.38%**, within a 1% allowance for figure pixel/axis digitization. This is now a regression test in `tests/test_publication_figure_consistency.py`.

This confirms the arithmetic/weighting path against a figure-derived spectrum and the paper's summary table; it does **not** validate detector calibration or replace the original event data.

## Evidence boundary

The original IR-T1 raw diagnostic files are not included with these papers and are not assumed to be publicly available. The first reproduction pass therefore must label each reference as one of:
- **Raw-data reproduction** — original calibrated input data and acquisition metadata available.
- **Figure-derived consistency test** — input or output values digitized from a published figure, with figure/panel and digitization uncertainty recorded.
- **Synthetic algorithm test** — known constructed signals verify numerical behavior but do not validate the experimental result.

Do not label the latter two as recovery of original experimental measurements. If a paper's text and graph conflict, retain both values and the source locations; use a clearly identified graph-derived value as the provisional target only when the curve and axes are unambiguous.

## Current implementation checkpoints

- Generic spectral fitting, wavelength calibration, conditional Doppler estimates, peak detection and reference-line matching: implemented. Species matches can be ambiguous; a user-supplied validated line list is required, and full atomic population inference is not implemented.
- Langmuir I–V first-pass model fitting: implemented; a compound/Gundestrup probe's full collection model is not implied.
- Welch PSD, FFT and cross-channel analysis: implemented.
- SVD channel decomposition and Morlet CWT: implemented in the current development branch with synthetic numerical regression tests.
- Explicit (E\times B) drift, instantaneous particle-flux products, Reynolds-stress product and separate fluctuation covariances: implemented with sign conventions required from the analyst.
- Reference-result comparison: scalar values and full curves/time series can be compared; the workflow uses supplied uncertainties/tolerances, reports error metrics and refuses to invent an agreement threshold.
- Baseline/treatment comparison: mean, RMS, and paired time-window changes are calculated for limiter/pressure conditions; unstable mean-percent changes are withheld near zero.

**Still required for paper-level confirmation:** digitize the chosen plotted targets, preserve source coordinates and digitization uncertainty, run the matching analysis on the appropriate input (where available), and export a comparison table. A figure containing only an already-processed result is not a substitute for raw diagnostic input.
