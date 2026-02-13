"""
Diagnostics module for Strain MCMC Inference Pipeline
Post-processing, convergence diagnostics, and derived parameters
"""

import os
import numpy as np
import emcee
from emcee.backends import HDFBackend
from scipy.stats import rankdata, norm
from numpy.typing import NDArray
from typing import cast
from strain_config import CFG, HDF5_DIR, theta_inits, MNU_EV
from strain_likelihoods import chi2_total
from strain_priors import log_prior, log_probability

# ============================================================================
# POST-RUN ANALYSIS UTILITIES (MULTI-CHAIN)
# ============================================================================

def load_chain(h5_path):
    backend = HDFBackend(h5_path, read_only=True)
    chain = backend.get_chain(flat=False)
    logp  = backend.get_log_prob(flat=False)
    return chain, logp

def flatten_chain(chain, discard=0, thin=1):
    chain = chain[discard::thin]
    nsteps, nwalkers, ndim = chain.shape
    return chain.reshape(nsteps * nwalkers, ndim)

def summarize_posterior_multichain(chains_flat, labels=("Ωmeff", "ωb", "ωc", "Ωinh0")):
    samples = np.vstack(chains_flat)

    print("\n" + "=" * 60)
    print("COMBINED POSTERIOR SUMMARY (ALL CHAINS)")
    print("=" * 60)

    for i, lab in enumerate(labels):
        s = samples[:, i]
        mean = np.mean(s)
        median = np.median(s)
        lo, hi = np.percentile(s, [16, 84])
        print(f"{lab:6s}: mean={mean:.4f}, median={median:.4f}, 68%=[{lo:.4f}, {hi:.4f}]")

    return samples

def _rhat_basic(chains):

    """
    Vehtari et al. (2019) rank-normalized split R-hat
    Correct for emcee ensemble samplers:
      - each RUN is a chain
      - walkers are flattened into draws
    chains_raw: list of arrays (nsteps, nwalkers, ndim)
    """
    nchains, ndraws = chains.shape

    chain_means = np.mean(chains, axis=1)
    chain_vars  = np.var(chains, axis=1, ddof=1)

    W = np.mean(chain_vars)
    B = ndraws * np.var(chain_means, ddof=1)

    var_hat = (ndraws - 1)/ndraws * W + B/ndraws
    return np.sqrt(var_hat / W)

def gelman_rubin_vehtari(chains_raw, param_names=None):

    nruns = len(chains_raw)
    nsteps, nwalkers, ndim = chains_raw[0].shape

    if param_names is None:
        param_names = [f"param_{i}" for i in range(ndim)]

    results = {}

    for i, name in enumerate(param_names):

        chains = [
            chains_raw[r][:, :, i].reshape(-1)
            for r in range(nruns)
        ]

        min_len = min(len(c) for c in chains)
        chains = [c[:min_len] for c in chains]

        chains = np.asarray(chains)

        ranks = rankdata(chains.ravel(), method="average")
        z = norm.ppf((ranks - 0.5) / len(ranks))
        z = z.reshape(chains.shape)

        z2 = cast(NDArray[np.floating], z)

        half: int = z2.shape[1] // 2
        z_split = np.vstack([z2[:, :half], z2[:, half:2 * half]])
        
        rhat_bulk = _rhat_basic(z_split)

        med = np.median(chains)
        folded = np.abs(chains - med)
        ranks_f = rankdata(folded.ravel(), method="average")
        zf = norm.ppf((ranks_f - 0.5) / len(ranks_f))
        zf = zf.reshape(chains.shape)
        zf_split = np.vstack([zf[:, :half], zf[:, half:2*half]])

        rhat_tail = _rhat_basic(zf_split)

        results[name] = {
            "rhat_bulk": rhat_bulk,
            "rhat_tail": rhat_tail,
            "rhat": max(rhat_bulk, rhat_tail),
        }

    return results

def global_map(chains_raw, logps_raw):
    best_lp = -np.inf
    best_theta = None

    for chain, logp in zip(chains_raw, logps_raw):
        chain_flat = chain.reshape(-1, chain.shape[-1])
        logp_flat  = logp.reshape(-1)

        idx = np.argmax(logp_flat)

        if logp_flat[idx] > best_lp:
            best_lp = logp_flat[idx]
            best_theta = chain_flat[idx]

    return best_theta, chi2_total(best_theta)

def effective_sample_size_emcee(chains_raw, tol=0):
    ndim = chains_raw[0].shape[-1]

    chains_concat = np.concatenate(chains_raw, axis=0)
    nsteps, nwalkers, _ = chains_concat.shape

    try:
        tau = emcee.autocorr.integrated_time(
            chains_concat,
            tol=tol,
            quiet=True
        )
    except Exception:
        tau = np.full(ndim, np.nan)

    N_total = nsteps * nwalkers
    ess = N_total / tau

    return ess, tau

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

# ======================================================================
# FAST DIC computation (no likelihood reevaluation)
# ======================================================================

def compute_dic_from_logp(samples, logp, chi2_at_theta_bar):
  
    mask = np.isfinite(logp)
    logp = logp[mask]

    chi2_vals = -2.0 * logp

    D_bar = np.mean(chi2_vals)
    D_hat = chi2_at_theta_bar

    p_D = D_bar - D_hat
    DIC = D_bar + p_D

    return {
        "D_bar": D_bar,
        "D_hat": D_hat,
        "p_D": p_D,
        "DIC": DIC,
    }

# ======================================================================
# DERIVED PARAMETER: H0 POSTERIOR (STRAIN MODEL)
# ======================================================================

def derive_h0_early_samples(samples, mnu_eV=MNU_EV):
    Om_m = np.atleast_1d(samples[:, 0])
    ombh2 = np.atleast_1d(samples[:, 1])
    omch2 = np.atleast_1d(samples[:, 2])
    omega_inh = np.atleast_1d(samples[:, 3])

    Om_m_ini = Om_m + omega_inh
    omega_nu = mnu_eV / 93.14
    h2 = (ombh2 + omch2 + omega_nu) / Om_m_ini
    h2 = np.atleast_1d(h2).astype(np.float64)

    H0 = np.full(len(h2), np.nan, dtype=np.float64)
    mask = h2 > 0
    H0[mask] = 100.0 * np.sqrt(h2[mask])
    return H0

def derive_h0_late_samples(samples, mnu_eV=MNU_EV):
    Om_m = np.atleast_1d(samples[:, 0])
    ombh2 = np.atleast_1d(samples[:, 1])
    omch2 = np.atleast_1d(samples[:, 2])

    omega_nu = mnu_eV / 93.14

    h2 = (ombh2 + omch2 + omega_nu) / Om_m
    h2 = np.atleast_1d(h2).astype(np.float64)

    H0 = np.full(len(h2), np.nan, dtype=np.float64)
    mask = h2 > 0.0
    H0[mask] = 100.0 * np.sqrt(h2[mask])
    return H0

def summarize_1d(x):
    mean = np.mean(x)
    median = np.median(x)
    lo, hi = np.percentile(x, [16, 84])
    return mean, median, lo, hi
