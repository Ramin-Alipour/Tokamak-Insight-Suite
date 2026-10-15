# TCV L-H threshold scaling reproducibility

## Source and method

- Dataset: [Zenodo record 14996664](https://zenodo.org/records/14996664), file `lhdatabase.h5`.
- Article: DOI [10.1088/1361-6587/adc8ce](https://doi.org/10.1088/1361-6587/adc8ce).
- Reproduced threshold scaling used by the published notebook:

[
P_{\mathrm{ITPA}}=0.049\,|\bar n_{e,20}|^{0.72}|B_T|^{0.8}S_{\mathrm{plasma}}^{0.94}\quad (\mathrm{MW}).
]

The implementation reproduces the ten Figure-4 selection masks and calculates (P_{\mathrm{loss}}/P_{\mathrm{ITPA}}) and normalized uncertainty (\sigma_{P}/P_{\mathrm{ITPA}}).

## Result

**Formula and selection reproducibility: PASS.**

- 92 database rows downloaded and checked.
- All 92 have finite positive (P_{\mathrm{ITPA}}).
- 89 rows have finite loss-power values and uncertainties and are included in normalized-power statistics.
- Shots 68719, 69891 and 69892 are excluded from normalized-power ratios because `PLMW` and/or `PLMWSTD` is missing; their threshold prediction itself remains computable.
- Ten Figure-4 groups were reproduced from the notebook's masks. Groups with zero matching rows remain empty; no synthetic values were inserted.

| Figure-4 group | Valid points | Median (P_{\mathrm{loss}}/P_{\mathrm{ITPA}}) |
|---|---:|---:|
| Panel a: D, ILH, ABEAM=2 | 6 | 1.241 |
| Panel a: H, ILH, ABEAM=1 | 5 | 2.055 |
| Panel b: D, ILH, ABEAM=2 | 12 | 1.137 |
| Panel b: H, ILH, ABEAM=1 | 4 | 1.809 |

The remaining six groups have no matching records under the published masks in this 92-row database. This is a reproducibility check of the published database normalization, not a new analysis of raw diagnostic signals.

## Reproduction

```bash
python tools/validate_tcv_lh_threshold.py
```


## Independent Figure-2 power-balance consistency check

The same Zenodo record's `LH_DATA.h5` contains the Figure-2 `PLMW` and `PSEP` series. Applying the notebook's `PLMW > 0.2 MW` selection gives 78 finite paired points.

| Statistic | Recomputed value |
|---|---:|
| Pearson correlation | 0.99499 |
| Spearman correlation | 0.99350 |
| Linear-fit slope (P_{sep}) vs (P_{loss}) | 0.98015 |
| Linear-fit intercept | −0.04237 MW |
| Median (P_{sep}/P_{loss}) | 0.91908 |
| Median absolute difference | 0.04990 MW |
| 90th-percentile absolute difference | 0.07240 MW |

The prespecified consistency thresholds (correlation at least 0.98, slope 0.90–1.10, and median ratio 0.85–1.15) all pass. The data show a small systematic offset despite the strong correlation; this is retained in the result rather than forced to the identity line.
