# EAST shot 22010 empirical validation

## Result

**Status: PASS — independent raw-data reproduction of published particle-flux statistics.**

- Experimental data: [Zenodo record 18473625](https://zenodo.org/records/18473625), files `22010ne.txt`, `22010vf2.txt`, and `22010vf3.txt`.
- Reference: Pécseli et al., *Small scale turbulence and plasma transport in the EAST tokamak*, DOI [10.1088/1361-6587/ae4f20](https://doi.org/10.1088/1361-6587/ae4f20).
- Published stationary interval: samples 275000–309600 inclusive, sampled at 2 MHz.
- Published geometry: potential-probe separation 8 mm; magnetic field near the LCFS 1.8 T.
- The archived density series is converted by (10^{19}\,\mathrm{m}^{-3}) per file unit. This scale is independently checked against the paper's reported density-fluctuation standard deviation, not selected to fit the flux mean.

## Method

Each of the three measured series is detrended independently using a fourth-order polynomial over the published stationary interval. The fluctuating density and electric field are then combined as

\[
E_{23}(t)=\frac{\widetilde{v}_{f3}(t)-\widetilde{v}_{f2}(t)}{0.008\;\mathrm{m}},
\qquad
\Gamma(t)=\frac{\widetilde{n}_e(t)E_{23}(t)}{1.8\;\mathrm{T}}.
\]

The computed mean and sample standard deviation of \(\Gamma\) are compared with the paper's values. Independent checks of density and potential fluctuation levels, flux skewness, and Pearson kurtosis are also applied. Tolerances are fixed in the validation script before comparing the final flux statistics.

## Numerical comparison

| Observable | Recomputed | Published | Relative difference |
|---|---:|---:|---:|
| Mean particle flux (m⁻² s⁻¹) | 1.31260 × 10²⁰ | 1.30 × 10²⁰ | 0.969% |
| Particle-flux standard deviation (m⁻² s⁻¹) | 2.63324 × 10²⁰ | 2.60 × 10²⁰ | 1.278% |
| Density fluctuation standard deviation (m⁻³) | 3.96092 × 10¹⁷ | 3.90 × 10¹⁷ | 1.562% |
| (v_{f2}) fluctuation standard deviation (V) | 7.90947 | 7.9 | 0.120% |
| (v_{f3}) fluctuation standard deviation (V) | 6.80548 | 6.8 | 0.081% |
| Flux skewness | 2.48058 | 2.5 | 0.777% |
| Flux Pearson kurtosis | 13.1894 | 13.5 | 2.301% |

All seven prespecified numerical checks pass. The run also produced the sample-by-sample CSV, JSON result, raw downloaded input files, and SHA-256 provenance records as a GitHub Actions artifact.

## Reproduction

Run:

```bash
python tools/validate_east_22010_flux.py --output-dir artifacts/east-22010-validation
```

The automated workflow `.github/workflows/east-public-data-validation.yml` executes this same canonical script. The result validates this particular measured-data analysis path and its stated preprocessing; it is not a blanket validation of every diagnostic or analysis method in the suite.
