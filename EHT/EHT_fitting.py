"""
Chi-squared fitting module for SBGEFT Black Hole Shadow MCMC Analysis
Grid-based chi-squared fitting for M87* and Sgr A* observations
"""

import numpy as np
from EHT_helpers import theta_ratio


# ============================================================
# χ² grid
# ============================================================
c4_grid = np.linspace(0, 3, 1501)


def chi2_curve(theta_GR, obs, sig):
    chi2 = []
    for c4 in c4_grid:
        rfac = theta_ratio(c4)
        if not np.isfinite(rfac):
            chi2.append(np.inf)
            continue
        th = theta_GR * rfac
        chi2.append((obs - th) ** 2 / (sig ** 2))
    return np.array(chi2)


def summarize(label, chi2):
    valid = np.isfinite(chi2)
    chi2_valid = chi2[valid]
    c4_valid = c4_grid[valid]


    i = int(np.argmin(chi2_valid))
    c4_best = float(c4_valid[i])
    chi2_min = float(chi2_valid[i])


    target = chi2_min + 1.0
    upper = None
    for j in range(i, len(c4_valid)):
        if chi2_valid[j] > target:
            upper = float(c4_valid[j])
            break


    print(label)
    print(f"  best c4 = {c4_best:.3f}")
    print(f"  chi2_min = {chi2_min:.3e}")
    if upper is None:
        print(f"  1σ upper > {c4_grid[-1]:.2f} (range limit)\n")
    else:
        print(f"  1σ upper = {upper:.3f}\n")
