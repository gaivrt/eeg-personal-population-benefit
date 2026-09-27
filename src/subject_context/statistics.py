import numpy as np
from scipy.stats import wilcoxon, PermutationMethod


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    assert np.all(np.isfinite(p)) and np.all((0 <= p) & (p <= 1))
    order = np.argsort(p, kind="stable")
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return result


def paired_statistics(ours, reference):
    a, b = np.asarray(ours), np.asarray(reference)
    assert a.shape == b.shape and a.ndim == 1 and len(a)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    d = np.round(a - b, 12)
    nonzero = d[d != 0]
    if not len(nonzero):
        statistic, p = 0., 1.
        method = "all_zero"
    else:
        tied = len(set(abs(nonzero))) != len(nonzero)
        method = "exact" if not tied else "permutation_65536_seed0"
        option = "exact" if not tied else PermutationMethod(n_resamples=65536, random_state=0)
        statistic, p = wilcoxon(nonzero, alternative="two-sided", method=option)
    return {"n_subjects": len(d), "n_nonzero": len(nonzero), "statistic": float(statistic),
            "p_raw": float(p), "method": method, "median_difference": float(np.median(d)),
            "improved_fraction": float(np.mean(d > 0)), "drop_gt_2pp_fraction": float(np.mean(d < -.02))}


def equivalent_trials(score, n, accuracies):
    """Only interpolate an increasing observed B5 curve. Never extrapolate."""
    n, a = np.asarray(n), np.asarray(accuracies)
    if len(n) < 2 or np.any(np.diff(n) <= 0) or np.any(np.diff(a) <= 0):
        return None
    if not a[0] <= score <= a[-1]:
        return None
    return float(np.interp(score, a, n))
