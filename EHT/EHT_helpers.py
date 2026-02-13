"""
Helper functions for SBGEFT Black Hole Shadow MCMC Analysis
Angular size calculations and shadow size ratios
"""

import numpy as np
from EHT_config import G, c, rad_to_uas, b0_dim
from EHT_physics import bcrit


def rg(M):
    return G * M / c ** 2


def theta_schwarzschild(M, D):
    return (2 * b0_dim * rg(M) / D) * rad_to_uas


def kerr_size_factor(a):
    return 1 - 0.041 * a * a


def theta_kerr(M, D, a):
    return theta_schwarzschild(M, D) * kerr_size_factor(a)


def theta_ratio(c4):
    bc_val, _ = bcrit(c4)
    if not np.isfinite(bc_val):
        return np.nan
    return bc_val / b0_dim

