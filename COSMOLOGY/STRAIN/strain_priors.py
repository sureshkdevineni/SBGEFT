"""
Prior and probability functions for Strain MCMC Inference Pipeline
Defines parameter priors and log-probability
"""

import numpy as np
from strain_config import CFG, MNU_EV
from strain_cosmology import derive_H0
from strain_likelihoods import chi2_total

# ============================================================================
# Priors, objective
# ============================================================================
def log_prior(theta):

    if len(theta) != 4:
        return -np.inf

    Om_m_eff, ombh2, omch2, Omega_inh0 = theta
    if not (0.1 < Om_m_eff < 0.5): return -np.inf
    if not (0.019 < ombh2 < 0.025): return -np.inf
    if not (0.06 < omch2 < 0.18): return -np.inf
    lo, hi = CFG.get('Omega_inh_bounds', (0, 0.3))
    if not (lo <= Omega_inh0 <= hi): return -np.inf

    H0 = derive_H0(Om_m_eff, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None or not (55 < H0 < 85):
        return -np.inf

    lp = 0.0

    bbn = CFG.get('BBN_prior', {})
    if bbn.get('use', False):
        lp += -0.5*((ombh2 - bbn['mean'])/bbn['sigma'])**2
    H0p = CFG.get('H0_prior', {})
    if H0p.get('use', False):
        lp += -0.5*((H0 - H0p['mean'])/H0p['sigma'])**2

    return lp

def log_likelihood(theta):
    chi2 = chi2_total(theta)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5 * chi2_total(theta)
