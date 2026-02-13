"""
Diagnostics module
Contains post-processing functions for convergence, model comparison, and summary statistics
"""

import os
import numpy as np
from numpy.typing import NDArray
from typing import cast
import emcee
from emcee.backends import HDFBackend
from scipy.stats import rankdata, norm
from LCDM_likelihoods import chi2_total


def load_chain(h5_path):
    backend = HDFBackend(h5_path, read_only=True)
    chain = backend.get_chain(flat=False)
    logp  = backend.get_log_prob(flat=False)
    return chain, logp


def flatten_chain(chain, discard=0, thin=1):
    chain = chain[discard::thin]
    nsteps, nwalkers, ndim = chain.shape
    return chain.reshape(nsteps * nwalkers, ndim)


def summarize_posterior_multichain(chains_flat, labels=("Ωm", "ωb", "ωc")):
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
    nchains, ndraws = chains.shape

    chain_means = np.mean(chains, axis=1)
    chain_vars  = np.var(chains, axis=1, ddof=1)

    W = np.mean(chain_vars)
    B = ndraws * np.var(chain_means, ddof=1)

    var_hat = (ndraws - 1)/ndraws * W + B/ndraws
    return np.sqrt(var_hat / W)


def gelman_rubin_vehtari(chains_raw, param_names=None):
    """
    Vehtari et al. (2019) rank-normalized split R-hat
    Correct for emcee ensemble samplers:
      - each RUN is a chain
      - walkers are flattened into draws
    chains_raw: list of arrays (nsteps, nwalkers, ndim)
    """

    nruns = len(chains_raw)
    nsteps, nwalkers, ndim = chains_raw[0].shape

    if param_names is None:
        param_names = [f"param_{i}" for i in range(ndim)]

    results = {}

    for i, name in enumerate(param_names):

        # --- flatten walkers inside each run
        chains = [
            chains_raw[r][:, :, i].reshape(-1)
            for r in range(nruns)
        ]  # list of 1D arrays

        # ensure equal length
        min_len = min(len(c) for c in chains)
        chains = [c[:min_len] for c in chains]

        # shape: (nchains, ndraws)
        chains = np.asarray(chains)

        # ---------- rank normalization ----------
        ranks = rankdata(chains.ravel(), method="average")
        z = norm.ppf((ranks - 0.5) / len(ranks))
        z = z.reshape(chains.shape)

        # ---------- split chains ----------
        z2 = cast(NDArray[np.floating], z)

        half: int = z2.shape[1] // 2
        z_split = np.vstack([z2[:, :half], z2[:, half:2 * half]])

        # ---------- bulk R-hat ----------
        rhat_bulk = _rhat_basic(z_split)

        # ---------- tail R-hat ----------
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

    # concatenate runs along steps
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


def compute_dic_from_logp(samples, logp, chi2_at_theta_bar):

    # Remove non-finite entries
    mask = np.isfinite(logp)
    logp = logp[mask]

    # Recover chi^2 (up to additive prior constant, which cancels in DIC)
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


def report_counts():
    from LCDM_config import CFG
    from LCDM_likelihoods import y_bao, m_obs_pantheon, mu_des5y
    
    n_bao = int(y_bao.size) if CFG.get('use_bao', True) else 0
    n_snP = len(m_obs_pantheon) if CFG.get('use_pantheon_sn', False) else 0
    n_snD = len(mu_des5y) if CFG.get('use_des5y_sn', False) else 0

    print(f"[DATA] BAO points used:      {n_bao}")
    print(f"[DATA] Pantheon+ SNe used:   {n_snP}")
    print(f"[DATA] DES-SN5YR SNe used:   {n_snD}")
    print(f"[DATA] CMB priors used:      {CFG.get('use_cmb_priors', False)}")
    print(f"[DATA] H0 prior used:        {CFG.get('H0_prior', {}).get('use', False)}")
    print(f"[DATA] BBN prior used:       {CFG.get('BBN_prior', {}).get('use', False)}")
