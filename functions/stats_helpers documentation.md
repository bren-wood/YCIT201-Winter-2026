# `stats_helpers` Documentation

A lightweight collection of statistics helper functions for teaching and quick analyses.

**Focus areas:** CLT simulation, z/t values, p-values, confidence intervals, margins of error, proportions, sample sizes, and one/two-sample tests.

- Pure-Python implementations where possible (Normal CDF/PPF via Acklam).
- **SciPy required** for Student’s *t* CDF/PPF (and any function that uses those).
- **NumPy required** for CLT simulation and two-sample *t* test inputs.

------

## Installation & Requirements

\# Required for some functions (CLT simulation, arrays)

`pip install numpy`

\# Required for Student-t quantiles/CDF and any function using them

`pip install scipy`

- If SciPy is missing and you call a function that needs it (e.g., `t_ppf`, `ci_mean_unknown_sigma`, or `one_sample_t_test`), you’ll get a helpful `ImportError`.
- If NumPy is missing and you call a function that needs it (e.g., `clt_sample_means`, `two_sample_t_test`), you’ll get a helpful `ImportError`.

------

## Quick Start


```python
from stats_helpers import *

\# Z-based CI for a mean when population sigma is known

ci = ci_mean_known_sigma(xbar=102.4, sigma=15, n=50, confidence=0.95)

\# t-based CI for a mean when sigma is unknown (requires SciPy)

ci_t = ci_mean_unknown_sigma(xbar=102.4, s=16.2, n=50, confidence=0.95)

\# One-sample z-test (known sigma)

z, p = one_sample_z_test(xbar=102.4, mu0=100, sigma=15, n=50, alternative="two-sided")

\# Two-proportion z test

z2, p2 = two_proportion_z_test(x1_success=42, n1=100, x2_success=55, n2=120, alternative="greater")

\# Sample size for mean (target margin of error)

n_req = sample_size_mean(moe=3, sigma=12, confidence=0.95)

\# CLT simulation of sample means

import numpy as np

rng = np.random.default_rng(0)

sampler = lambda k: rng.exponential(scale=1.0, size=k) # non-normal population

means = clt_sample_means(sampler, n=30, reps=5000)    # sample means approx normal
```


------

## Quick Reference (Cheat Sheet)

### Z & Normal

```python
z = z_score(x=observed, mu=mean0, sigma=sd)      # (x - mu)/sigma

p = p_value_z(z, alternative="two-sided|less|greater") # from standard normal

zcrit = z_critical(confidence=0.95)          # N(0,1) critical value
```

### t & Student’s t (SciPy required)

```python
t = t_score(xbar, mu0, s, n)              # (xbar - mu0)/(s/sqrt(n))

p = p_value_t(t, df=n-1, alternative="two-sided|...") # from Student’s t

tcrit = t_critical(df=n-1, confidence=0.95)      # t critical value
```

### Confidence Intervals

```python
ci_mean_known_sigma(xbar, sigma, n, confidence=0.95)

ci_mean_unknown_sigma(xbar, s, n, confidence=0.95)   # SciPy required

ci_proportion_wald(phat, n, confidence=0.95)      # simple, less robust

ci_proportion_wilson(phat, n, confidence=0.95)     # recommended for small n or extreme phat
```

### Margins of Error

```python
margin_of_error_mean_known_sigma(sigma, n, confidence=0.95)

margin_of_error_mean_unknown_sigma(s, n, confidence=0.95) # SciPy required

margin_of_error_proportion(phat, n, confidence=0.95)
```

### Sample Sizes

```python
sample_size_mean(moe, sigma, confidence=0.95)

sample_size_proportion(moe, p_guess=0.5, confidence=0.95) # p_guess=0.5 is conservative
```

### Hypothesis Tests

```python
one_sample_z_test(xbar, mu0, sigma, n, alternative="two-sided")

one_sample_t_test(xbar, mu0, s, n, alternative="two-sided") # SciPy required

two_sample_t_test(x1, x2, equal_var=False, alternative="two-sided") # Welch by default, NumPy+SciPy required

two_proportion_z_test(x1_success, n1, x2_success, n2, alternative="two-sided")
```

### CLT Simulation

clt_sample_means(sampler, n, reps=1000) # sampler: function that returns k IID draws

------

## API Reference & Examples

### Normal Distribution Helpers (No SciPy Required)

#### `normal_cdf(x, mu=0.0, sigma=1.0) -> float`

Cumulative probability $P(X \le x)$ for $X \sim \mathcal{N}(\mu, \sigma^2)$.


```python
p = normal_cdf(1.96)        # ~0.975

p_std = normal_cdf(70, mu=65, sigma=5) # ~0.977
```


#### `normal_ppf(p, mu=0.0, sigma=1.0) -> float`

Quantile (inverse CDF). Uses Acklam’s approximation with a refinement step.


```python
q975 = normal_ppf(0.975)      # ~1.96 (z* for 95% two-sided CI)

xq = normal_ppf(0.90, mu=100, sigma=15) # 90th percentile
```


> **Tip:** For a two-sided 95% CI, use `normal_ppf(1 - 0.05/2)`.

------

### Student-t Distribution (SciPy Required)

#### `t_cdf(x, df) -> float` and `t_ppf(p, df) -> float`


```python
from stats_helpers import t_ppf

tcrit = t_ppf(0.975, df=24)  # two-sided 95% t critical
```


------

### Standard Errors & Test Statistics

#### `se_mean_known_sigma(sigma, n)` and `se_mean_unknown_sigma(s, n)`


```python
se_z = se_mean_known_sigma(sigma=12, n=36) # 12 / 6 = 2

se_t = se_mean_unknown_sigma(s=13.5, n=40)

```

#### `z_score(x, mu, sigma)` and `t_score(xbar, mu0, s, n)`


```python
z = z_score(x=105, mu=100, sigma=10)        # 0.5

t = t_score(xbar=102.4, mu0=100, s=16.2, n=50)
```


------

### p-Values

#### `p_value_z(z, alternative="two-sided")`

p = p_value_z(2.1, alternative="greater")  # right tail

#### `p_value_t(t, df, alternative="two-sided")` (SciPy required)

p = p_value_t(-1.85, df=24, alternative="less")

------

### Confidence Intervals & Margins of Error

#### Mean with Known σ



ci = ci_mean_known_sigma(xbar=72, sigma=8, n=64, confidence=0.95)

\# (72 - z* * 8/sqrt(64), 72 + z* * 8/sqrt(64))



#### Mean with Unknown σ (t-based; SciPy required)

ci = ci_mean_unknown_sigma(xbar=72, s=8.5, n=25, confidence=0.95)

#### Proportion CI (Wald vs Wilson)



phat, n = 18/50, 50

wald = ci_proportion_wald(phat, n, confidence=0.95)

wilson = ci_proportion_wilson(phat, n, confidence=0.95) # more reliable near 0/1 or small n



#### Margin of Error (MoE)



moe_mean_z = margin_of_error_mean_known_sigma(sigma=10, n=100, confidence=0.95)

moe_prop = margin_of_error_proportion(phat=0.38, n=200, confidence=0.95)

``



------

### Sample Size Calculators

#### For Mean (Z-based)



n_req = sample_size_mean(moe=2.5, sigma=12, confidence=0.95)

``



#### For Proportion

n_req = sample_size_proportion(moe=0.04, p_guess=0.5, confidence=0.95) # conservative

------

### One-/Two-Sample Tests

#### One-Sample Z-Test (Known σ)



z, p = one_sample_z_test(xbar=102.4, mu0=100, sigma=15, n=50, alternative="two-sided")

\# Decision at alpha = 0.05: reject H0 if p <= 0.05



#### One-Sample t-Test (Unknown σ; SciPy required)



t, p, df = one_sample_t_test(xbar=5.3, mu0=5.0, s=1.1, n=36, alternative="greater")

``



#### Two-Sample t-Test (Welch by default; NumPy + SciPy required)



import numpy as np

x1 = np.array([12.1, 11.8, 12.4, 12.0, 11.9])

x2 = np.array([11.3, 11.5, 11.1, 11.6, 11.2, 11.0])

t, p, df = two_sample_t_test(x1, x2, equal_var=False, alternative="two-sided")

``



- `equal_var=False` → Welch’s test (unequal variances), safer default.
- `equal_var=True` → Pooled-variance test.

#### Two-Proportion Z-Test



\# Group A: 42/100 successes; Group B: 55/120 successes. Is B > A?

z, p = two_proportion_z_test(42, 100, 55, 120, alternative="less") # "less" tests p1 < p2



------

### Central Limit Theorem Simulation

#### `clt_sample_means(sampler, n, reps=1000) -> np.ndarray`

Draw `reps` independent samples of size `n` from your `sampler(k)` and return their means.



import numpy as np

rng = np.random.default_rng(123)

\# Non-normal population (Exponential)

def sampler(k):

  return rng.exponential(scale=1.0, size=k) # mean=1, var=1

means_n10 = clt_sample_means(sampler, n=10, reps=5000)

means_n40 = clt_sample_means(sampler, n=40, reps=5000)

\# Plotting idea (in a notebook):

\# import matplotlib.pyplot as plt

\# plt.hist(means_n10, bins=40, density=True, alpha=0.6, label='n=10')

\# plt.hist(means_n40, bins=40, density=True, alpha=0.6, label='n=40')

\# plt.legend(); plt.title("CLT: Sample Means Approximate Normality")

``



------

## Common Workflows

### 1) CI vs Hypothesis Test (Mean, known σ)



\# CI approach

low, high = ci_mean_known_sigma(xbar=102.4, sigma=15, n=50, confidence=0.95)

\# If mu0=100 is outside [low, high], reject H0 at alpha=0.05.

\# NHST approach

z, p = one_sample_z_test(102.4, mu0=100, sigma=15, n=50)

\# If p <= 0.05, reject H0.



### 2) Unknown σ → Use t (requires SciPy)



ci = ci_mean_unknown_sigma(xbar=14.2, s=3.8, n=22, confidence=0.95)

t, p, df = one_sample_t_test(14.2, mu0=13.0, s=3.8, n=22, alternative="greater")



### 3) Comparing Two Groups (Means vs Proportions)

- Means → `two_sample_t_test` (Welch default)
- Proportions → `two_proportion_z_test` (with pooled SE)

------

## Notes, Pitfalls, and Teaching Tips

- **Alternative hypothesis** must match the research question:
  - `"two-sided"` tests differences in either direction.
  - `"greater"` tests (Group A > Group B) or (μ > μ₀).
  - `"less"` tests (Group A < Group B) or (μ < μ₀).
- **Wald CI for proportions** can be poor for small `n` or extreme `phat`; prefer **Wilson** in those cases.
- **Known σ vs Unknown σ**:
  - Use **z** only when population σ is known (rare in practice) or when justified by large `n` with a reliable σ estimate.
  - Otherwise, use **t** (requires SciPy for quantiles/p-values).
- **Welch’s t-test** (`equal_var=False`) is safer than pooled unless you have strong evidence of equal variances.
- **CLT simulations** are great to **visually demonstrate** that sample means approach normality even from skewed populations.
- **Sample size calculators** give **minimum** `n`; round up. For proportions, `p_guess=0.5` yields the **most conservative** (largest) `n`.

------

## Minimal Test (Sanity Check)



python stats_helpers.py

\# Expected: "stats_helpers: basic checks passed."



------

## License & Attribution

- Normal PPF uses Acklam’s inverse-normal approximation (with a single Halley refinement).
- This module is intended for educational use and quick applied workflows.
