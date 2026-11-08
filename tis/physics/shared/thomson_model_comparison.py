"""Model comparison for calibrated Thomson forward spectra."""
import numpy as np


def compare_thomson_forward_models(measured_spectrum, model_spectra, parameter_counts,
                                   spectral_covariance=None, pointwise_uncertainty=None):
    """Compare completed candidate forward fits on identical calibrated bins."""
    y = np.asarray(measured_spectrum, dtype=float)
    if y.ndim != 1 or y.size < 3 or not np.isfinite(y).all():
        raise ValueError("measured_spectrum must be a finite vector with at least 3 samples")
    if len(model_spectra) < 2:
        raise ValueError("at least two candidate models are required")
    if spectral_covariance is not None and pointwise_uncertainty is not None:
        raise ValueError("supply only one uncertainty representation")
    n = y.size
    if spectral_covariance is not None:
        c = np.asarray(spectral_covariance, dtype=float)
        if c.shape != (n, n) or not np.isfinite(c).all() or not np.allclose(c, c.T):
            raise ValueError("spectral_covariance must be finite, symmetric and (n,n)")
        try:
            L = np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        whiten = lambda r: np.linalg.solve(L, r)
        logdet = 2 * np.log(np.diag(L)).sum()
        known = True
        noise = "supplied_full_Gaussian_covariance"
    elif pointwise_uncertainty is not None:
        s = np.asarray(pointwise_uncertainty, dtype=float)
        if s.shape != y.shape or not np.isfinite(s).all() or np.any(s <= 0):
            raise ValueError("pointwise_uncertainty must be positive and match measured_spectrum")
        whiten = lambda r: r / s
        logdet = 2 * np.log(s).sum()
        known = True
        noise = "supplied_pointwise_Gaussian_uncertainty"
    else:
        whiten = lambda r: r
        logdet = None
        known = False
        noise = "unweighted_common_Gaussian_variance_MLE"
    out = {}
    for name, prediction in model_spectra.items():
        if name not in parameter_counts:
            raise ValueError(f"missing parameter count for {name}")
        k = parameter_counts[name]
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError(f"invalid parameter count for {name}")
        p = np.asarray(prediction, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f"prediction for {name} must match measured_spectrum")
        q = float(whiten(y - p) @ whiten(y - p))
        ke = int(k) if known else int(k) + 1
        if known:
            dev = q + float(logdet) + n * np.log(2 * np.pi)
            var = None
        else:
            var = max(q / n, np.finfo(float).tiny)
            dev = n * (np.log(2 * np.pi * var) + 1)
        aic = float(dev + 2 * ke)
        bic = float(dev + ke * np.log(n))
        aicc = float(aic + 2 * ke * (ke + 1) / (n - ke - 1)) if n > ke + 1 else None
        out[name] = {"n_observations": n, "fitted_parameter_count": int(k),
                     "effective_parameter_count": ke, "weighted_residual_sum_squares": q,
                     "minus_two_log_likelihood": float(dev), "AIC": aic, "AICc": aicc,
                     "BIC": bic, "residual_variance_MLE": var, "noise_model": noise}
    for metric in ("AIC", "AICc", "BIC"):
        finite = [v[metric] for v in out.values() if v[metric] is not None]
        best = min(finite) if finite else None
        for value in out.values():
            value["delta_" + metric] = None if best is None or value[metric] is None else value[metric] - best
    return {"models": out, "ranking_metric": "AICc" if any(v["AICc"] is not None for v in out.values()) else "AIC",
            "noise_model": noise, "n_observations": n,
            "warning": "Information criteria compare fitted likelihoods; they do not establish physical truth or experimental validation."}
