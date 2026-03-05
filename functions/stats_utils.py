import numpy as np
from scipy import stats

# Ths is based on the alpha (a probability) and the p-value (another probability)
# When p < a reject null hypothesis.
# when p > a fail to reject the null hypothesis.
def perform_hyp_test_probability_verbose(a, p):
    print(f"Using alpha {a:.3f} and probability value {p:.3f}, we", end="")
    if p > a:
        print(" fail to", end="")
    print(" reject the null hypothesis.")

def perform_hyp_test_t_verbose(a, t):
    print(f"Using the t values, alpha {a:.3f} and t value {t:.3f}, we", end="")
    if t < a:
        print(" fail to", end="")
    print(" reject the null hypothesis.")


def calculate_t_value(d_bar, n, s_d):
    # Calculate the standard error
    standard_error = s_d / np.sqrt(n)
    
    # Calculate the t-value
    t_value = d_bar / standard_error
    
    return t_value
