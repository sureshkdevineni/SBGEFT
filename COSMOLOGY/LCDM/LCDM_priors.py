"""
Priors module
Contains prior definitions and probability functions for MCMC sampling
"""

import numpy as np
from LCDM_config import CFG, MNU_EV
from LCDM_cosmology import derive_H0
from LCDM_likelihoods import chi2_total


def log_prior(theta):
    if len(theta) != 3:
        return -np.inf
    Om_m, ombh2, omch2 = theta
    if not (0.1 < Om_m < 0.5): return -np.inf
    if not (0.019 < ombh2 < 0.025): return -np.inf
    if not (0.06 < omch2 < 0.18): return -np.inf

    H0 = derive_H0(Om_m, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None or not (55 < H0 < 82):
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


def prior_posterior_diagnostics(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        raise ValueError("log_prior is not finite at supplied theta")

    logpost = log_probability(theta)

    minus_logprior = -lp
    minus2_logpost = -2.0 * logpost

    print("\nPRIOR / POSTERIOR DIAGNOSTICS AT MAP")
    print("-" * 45)
    print(f"log prior           = {lp:.2f}")
    print(f"-log prior          = {minus_logprior:.2f}")
    print(f"2 * (-log prior)    = {2.0 * minus_logprior:.2f}")
    print(f"-2 log posterior    = {minus2_logpost:.2f}")
