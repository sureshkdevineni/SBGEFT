"""
MCMC inference module for SBGEFT Black Hole Shadow Analysis
Bayesian posterior estimation using emcee
"""

import numpy as np
import emcee
from EHT_config import M87_obs, M87_sig, SGR_obs, SGR_sig
from EHT_helpers import theta_ratio


def chi2_joint_s(c4, theta_M87_S, theta_SGR_S):
    rfac = theta_ratio(c4)
    if not np.isfinite(rfac):
        return np.inf
    th_M87 = theta_M87_S * rfac
    th_SGR = theta_SGR_S * rfac
    return ((M87_obs - th_M87) / M87_sig) ** 2 + ((SGR_obs - th_SGR) / SGR_sig) ** 2


def loglike_s(theta, theta_M87_S, theta_SGR_S):
    c4 = float(theta[0])
    if c4 < 0.0 or c4 > 3.0:
        return -np.inf
    return -0.5 * chi2_joint_s(c4, theta_M87_S, theta_SGR_S)


def chi2_joint_k(c4, theta_M87_K, theta_SGR_K):
    rfac = theta_ratio(c4)
    if not np.isfinite(rfac):
        return np.inf
    th_M87 = theta_M87_K * rfac
    th_SGR = theta_SGR_K * rfac
    return ((M87_obs - th_M87) / M87_sig) ** 2 + ((SGR_obs - th_SGR) / SGR_sig) ** 2


def loglike_k(theta, theta_M87_K, theta_SGR_K):
    c4 = float(theta[0])
    if c4 < 0.0 or c4 > 3.0:
        return -np.inf
    return -0.5 * chi2_joint_k(c4, theta_M87_K, theta_SGR_K)


def run_mcmc_schwarzschild(theta_M87_S, theta_SGR_S):
    """
    MCMC POSTERIOR: Schwarzschild (Joint)
    """
    print("=== (3) MCMC POSTERIOR (Joint Schwarzschild) ===")


    ndim, nwalkers, nsteps, burn = 1, 32, 4000, 1000
    p0 = 0.05 + 0.01 * np.random.randn(nwalkers, ndim)


    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike_s, args=(theta_M87_S, theta_SGR_S))
    sampler.run_mcmc(p0, nsteps, progress=True)


    chain = sampler.get_chain(discard=burn, flat=True)
    if chain is None:
        raise RuntimeError("sampler.get_chain returned None (unexpected).")
    c4_samples = chain[:, 0]


    median = float(np.median(c4_samples))
    lo, hi = np.percentile(c4_samples, [16, 84])


    print(f"Posterior median c4 = {median:.3f}")
    print(f"1σ interval         = [{lo:.3f}, {hi:.3f}]")
    print(f"Upper bound (84%)   ≈ {hi:.3f}\n")


def run_mcmc_kerr(theta_M87_K, theta_SGR_K):
    """
    MCMC POSTERIOR: Kerr (Joint)
    """
    print("=== (4) MCMC POSTERIOR (Joint Kerr) ===")


    ndim, nwalkers, nsteps, burn = 1, 32, 4000, 1000
    p0 = 0.05 + 0.01 * np.random.randn(nwalkers, ndim)


    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike_k, args=(theta_M87_K, theta_SGR_K))
    sampler.run_mcmc(p0, nsteps, progress=True)


    chain = sampler.get_chain(discard=burn, flat=True)
    if chain is None:
        raise RuntimeError("sampler.get_chain returned None (unexpected).")
    c4_samples = chain[:, 0]


    median = float(np.median(c4_samples))
    lo, hi = np.percentile(c4_samples, [16, 84])


    print(f"Posterior median c4 = {median:.3f}")
    print(f"1σ interval         = [{lo:.3f}, {hi:.3f}]")
    print(f"Upper bound (84%)   ≈ {hi:.3f}\n")
