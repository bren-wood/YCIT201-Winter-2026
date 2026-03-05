# stats_helpers.py
"""
A lightweight collection of statistics helper functions for teaching and quick analyses.

Focus areas: CLT simulation, z/t values, p-values, confidence intervals,
margin of error, proportions, and common one/two-sample tests.

- Pure-Python / NumPy implementations where feasible.
- Uses SciPy for Student-t CDF/PPF if available; otherwise raises a helpful error.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple, Optional

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

# --- Utilities ----------------------------------------------------------------

def _require_numpy():
    if np is None:
        raise ImportError("NumPy is required for this function. Please install numpy.")

# --- Normal distribution helpers (no SciPy required) ---------------------------
# Accurate inverse-normal approximation based on Peter John Acklam's algorithm.
# Reference: https://web.archive.org/web/20150910044751/http://home.online.no/~pjacklam/notes/invnorm/

_A = [
    -3.969683028665376e+01,
     2.209460984245205e+02,
    -2.759285104469687e+02,
     1.383577518672690e+02,
    -3.066479806614716e+01,
     2.506628277459239e+00
]
_B = [
    -5.447609879822406e+01,
     1.615858368580409e+02,
    -1.556989798598866e+02,
     6.680131188771972e+01,
    -1.328068155288572e+01
]
_C = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
     4.374664141464968e+00,
     2.938163982698783e+00
]
_D = [
     7.784695709041462e-03,
     3.224671290700398e-01,
     2.445134137142996e+00,
     3.754408661907416e+00
]

_P_LOW = 0.02425
_P_HIGH = 1 - _P_LOW

def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Cumulative distribution function for N(mu, sigma^2)."""
    z = (x - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Percent-point function (quantile) for N(mu, sigma^2) using Acklam approximation."""
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0,1)")

    # Transform to standard normal quantile
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3])
    elif p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
             (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3])

    # One iteration of Halley's method to improve accuracy
    e = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)

    return mu + sigma * x

# --- Student-t distribution (SciPy if available) -------------------------------

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

def t_cdf(x: float, df: int) -> float:
    """CDF of Student's t with df degrees of freedom. Requires SciPy."""
    if _scipy_stats is None:
        raise ImportError("t_cdf requires SciPy (scipy.stats). Please install scipy.")
    return _scipy_stats.t.cdf(x, df)

def t_ppf(p: float, df: int) -> float:
    """PPF (quantile) of Student's t with df degrees of freedom. Requires SciPy."""
    if _scipy_stats is None:
        raise ImportError("t_ppf requires SciPy (scipy.stats). Please install scipy.")
    return _scipy_stats.t.ppf(p, df)

# --- Standard errors, z/t statistics, and p-values -----------------------------

def se_mean_known_sigma(sigma: float, n: int) -> float:
    return sigma / math.sqrt(n)

def se_mean_unknown_sigma(s: float, n: int) -> float:
    return s / math.sqrt(n)

def z_score(x: float, mu: float, sigma: float) -> float:
    return (x - mu) / sigma

def t_score(xbar: float, mu0: float, s: float, n: int) -> float:
    return (xbar - mu0) / (s / math.sqrt(n))

def p_value_z(z: float, two_sided: bool = True, alternative: str = "two-sided") -> float:
    """p-value for a z statistic.
    alternative in {"two-sided","less","greater"}
    """
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    if alternative == "two-sided" or two_sided:
        return 2.0 * min(normal_cdf(z), 1 - normal_cdf(z))
    elif alternative == "less":
        return normal_cdf(z)
    else:
        return 1 - normal_cdf(z)

def p_value_t(t: float, df: int, alternative: str = "two-sided") -> float:
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    if _scipy_stats is None:
        raise ImportError("p_value_t requires SciPy (scipy.stats). Please install scipy.")
    if alternative == "two-sided":
        return 2.0 * min(_scipy_stats.t.cdf(t, df), 1 - _scipy_stats.t.cdf(t, df))
    elif alternative == "less":
        return _scipy_stats.t.cdf(t, df)
    else:
        return 1 - _scipy_stats.t.cdf(t, df)

# --- Confidence intervals and margins of error ---------------------------------

def ci_mean_known_sigma(xbar: float, sigma: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    z = normal_ppf(1 - alpha / 2)
    moe = z * se_mean_known_sigma(sigma, n)
    return xbar - moe, xbar + moe

def ci_mean_unknown_sigma(xbar: float, s: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    df = n - 1
    tcrit = t_ppf(1 - alpha / 2, df)
    moe = tcrit * se_mean_unknown_sigma(s, n)
    return xbar - moe, xbar + moe

def margin_of_error_mean_known_sigma(sigma: float, n: int, confidence: float = 0.95) -> float:
    z = normal_ppf(1 - (1 - confidence) / 2)
    return z * se_mean_known_sigma(sigma, n)

def margin_of_error_mean_unknown_sigma(s: float, n: int, confidence: float = 0.95) -> float:
    df = n - 1
    tcrit = t_ppf(1 - (1 - confidence) / 2, df)
    return tcrit * se_mean_unknown_sigma(s, n)

# Proportions
def ci_proportion_wald(phat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = normal_ppf(1 - (1 - confidence) / 2)
    se = math.sqrt(phat * (1 - phat) / n)
    moe = z * se
    return max(0.0, phat - moe), min(1.0, phat + moe)

def ci_proportion_wilson(phat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = normal_ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    adj = z * math.sqrt((phat*(1-phat)/n) + (z**2) / (4*n*n)) / denom
    return max(0.0, center - adj), min(1.0, center + adj)

def margin_of_error_proportion(phat: float, n: int, confidence: float = 0.95) -> float:
    z = normal_ppf(1 - (1 - confidence) / 2)
    return z * math.sqrt(phat * (1 - phat) / n)

# --- Sample size calculators ----------------------------------------------------

def sample_size_mean(moe: float, sigma: float, confidence: float = 0.95) -> int:
    z = normal_ppf(1 - (1 - confidence) / 2)
    n = (z * sigma / moe) ** 2
    return int(math.ceil(n))

def sample_size_proportion(moe: float, p_guess: float = 0.5, confidence: float = 0.95) -> int:
    z = normal_ppf(1 - (1 - confidence) / 2)
    n = (z**2) * p_guess * (1 - p_guess) / (moe**2)
    return int(math.ceil(n))

# --- One-/Two-sample tests -----------------------------------------------------

def one_sample_z_test(xbar: float, mu0: float, sigma: float, n: int, alternative: str = "two-sided") -> Tuple[float, float]:
    z = (xbar - mu0) / (sigma / math.sqrt(n))
    p = p_value_z(z, alternative=alternative)
    return z, p

def one_sample_t_test(xbar: float, mu0: float, s: float, n: int, alternative: str = "two-sided") -> Tuple[float, float, int]:
    df = n - 1
    t = (xbar - mu0) / (s / math.sqrt(n))
    p = p_value_t(t, df, alternative=alternative)
    return t, p, df

def two_sample_t_test(
    x1: Iterable[float],
    x2: Iterable[float],
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> Tuple[float, float, float]:
    """Welch's t-test by default. Returns (t, p, df)."""
    _require_numpy()
    x1 = np.asarray(list(x1), dtype=float)
    x2 = np.asarray(list(x2), dtype=float)
    n1, n2 = len(x1), len(x2)
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))

    if equal_var:
        sp2 = ((n1 - 1)*v1 + (n2 - 1)*v2) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        se = math.sqrt(v1/n1 + v2/n2)
        df = (v1/n1 + v2/n2)**2 / ((v1*v1)/((n1*n1)*(n1-1)) + (v2*v2)/((n2*n2)*(n2-1)))

    t = (m1 - m2) / se

    if _scipy_stats is None:
        raise ImportError("two_sample_t_test p-value requires SciPy; t and df are computed.")
    # Compute one-/two-sided p-value from t and df
    if alternative == "two-sided":
        p = 2.0 * min(_scipy_stats.t.cdf(t, df), 1 - _scipy_stats.t.cdf(t, df))
    elif alternative == "less":
        p = _scipy_stats.t.cdf(t, df)
    else:
        p = 1 - _scipy_stats.t.cdf(t, df)
    return t, p, df

def two_proportion_z_test(x1_success: int, n1: int, x2_success: int, n2: int, alternative: str = "two-sided") -> Tuple[float, float]:
    p1 = x1_success / n1
    p2 = x2_success / n2
    p_pool = (x1_success + x2_success) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p = p_value_z(z, alternative=alternative)
    return z, p

# --- Central Limit Theorem simulation -----------------------------------------

def clt_sample_means(
    sampler: Callable[[int], Iterable[float]],
    n: int,
    reps: int = 1000,
) -> "np.ndarray":
    """Draw `reps` samples of size n using `sampler(n)` and return their means.

    `sampler(k)` should return an iterable of k IID draws from the population of interest.
    """
    _require_numpy()
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = np.array(list(sampler(n)), dtype=float)
        means[i] = float(sample.mean())
    return means

# --- Convenience wrappers ------------------------------------------------------

def z_critical(confidence: float = 0.95) -> float:
    alpha = 1.0 - confidence
    return normal_ppf(1 - alpha/2)

def t_critical(df: int, confidence: float = 0.95) -> float:
    alpha = 1.0 - confidence
    return t_ppf(1 - alpha/2, df)


if __name__ == "__main__":
    # Quick self-checks
    assert abs(normal_cdf(0.0) - 0.5) < 1e-12
    # Symmetry check for ppf
    for p in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:
        q = normal_ppf(p)
        assert abs(normal_cdf(q) - p) < 1e-6
    print("stats_helpers: basic checks passed.")
