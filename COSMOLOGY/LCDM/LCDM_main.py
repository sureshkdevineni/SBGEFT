"""
Main analysis script
Runs multiple MCMC chains and performs comprehensive diagnostics
"""

import os
import numpy as np
from LCDM_config import CFG, HDF5_DIR, theta_inits
from LCDM_data_setup import setup_data
from LCDM_mcmc_sampler import run_mcmc, set_mcmc_files
from LCDM_diagnostics import (
    report_counts, load_chain, flatten_chain, summarize_posterior_multichain,
    gelman_rubin_vehtari, global_map, effective_sample_size_emcee,
    compute_dic_from_logp
)
from LCDM_priors import prior_posterior_diagnostics
from LCDM_likelihoods import chi2_decomposition, chi2_total
from LCDM_cosmology import derive_h0_samples
from LCDM_utils import reset_caches


def run_analysis():
    """
    Main analysis workflow: setup, run MCMC chains, and perform diagnostics
    """
    
    print("\n" + "="*70)
    print("RUNNING 4 INDEPENDENT ΛCDM MCMC CHAINS")
    print("="*70)

    # Setup data
    setup_data()
    report_counts()

    # ============================================================================
    # Run MCMC chains
    # ============================================================================
    for run_id, theta0 in theta_inits.items():

        print("\n" + "="*70)
        print(f"STARTING MCMC RUN {run_id}")
        print(f"Initial θ = {theta0}")
        print("="*70)

        # set run-specific initial point
        CFG['mcmc']['theta_init'] = theta0.tolist()

        # unique random seed per run
        CFG['mcmc']['seed'] = 12345 + run_id

        # unique HDF5 files per run
        set_mcmc_files(run_id)

        # clear caches for safety
        reset_caches()

        run_mcmc()

    print("\nALL FOUR MCMC RUNS COMPLETE ✅")

    # ============================================================================
    # POST-RUN DIAGNOSTICS (ALL CHAINS)
    # ============================================================================
    prod_paths = [
        os.path.join(HDF5_DIR, f"chain{i}_prod.h5")
        for i in theta_inits.keys()
    ]

    chains_raw = []
    logps_raw  = []

    for path in prod_paths:
        chain, logp = load_chain(path)
        chains_raw.append(chain)
        logps_raw.append(logp)

    # --- Posterior summary
    chains_flat = [flatten_chain(c) for c in chains_raw]
    samples = summarize_posterior_multichain(chains_flat)

    # --- Rhat
    rhat = gelman_rubin_vehtari(
        chains_raw,
        param_names=("Ωm", "ωb", "ωc")
    )

    print("\nVEHTARI GELMAN - RUBIN R̂")
    print("-" * 40)
    for p, v in rhat.items():
        print(
            f"{p:4s} : "
            f"R̂_bulk={v['rhat_bulk']:.4f}, "
            f"R̂_tail={v['rhat_tail']:.4f}, "
            f"R̂={v['rhat']:.4f}"
        )

    # --- Global MAP and χ²
    theta_map, chi2_map = global_map(chains_raw, logps_raw)
    print("\nGLOBAL MAP ESTIMATE")
    print("-" * 40)
    print(f"θ_MAP = {theta_map}")
    print(f"χ²_MAP = {chi2_map:.2f}")

    # --- Prior / posterior diagnostics at MAP
    prior_posterior_diagnostics(theta_map)

    # ============================================================================
    # DIC (FAST, SAFE)
    # ============================================================================
    logp_flat = np.concatenate([
        lp.reshape(-1) for lp in logps_raw
    ])

    dic = compute_dic_from_logp(
        samples=samples,
        logp=logp_flat,
        chi2_at_theta_bar=chi2_total(np.mean(samples, axis=0))
    )

    print("\n" + "=" * 60)
    print("DEVIANCE INFORMATION CRITERION (DIC)")
    print("=" * 60)
    print(f"D̄   (mean deviance)   = {dic['D_bar']:.2f}")
    print(f"D(θ̄)                 = {dic['D_hat']:.2f}")
    print(f"p_D (eff. parameters) = {dic['p_D']:.2f}")
    print(f"DIC                  = {dic['DIC']:.2f}")

    # --- χ² decomposition
    chi2s = chi2_decomposition(theta_map)
    print("\nχ² DECOMPOSITION AT MAP")
    print("-" * 40)
    for k, v in chi2s.items():
        print(f"{k:6s} : χ² = {v:.2f}")

    # --- Effective Sample Size (ESS)
    ess, tau = effective_sample_size_emcee(chains_raw)

    print("\nEFFECTIVE SAMPLE SIZE (ESS)")
    print("-" * 40)
    for lab, e, t in zip(("Ωm", "ωb", "ωc"), ess, tau):
        print(f"{lab:6s} : ESS ≈ {e:.4f}   (τ ≈ {t:.4f})")

    # ============================================================================
    # DERIVED PARAMETER: H0 POSTERIOR
    # ============================================================================
    H0_samples = derive_h0_samples(samples)
    H0_samples = H0_samples[np.isfinite(H0_samples)]

    H0_mean   = np.mean(H0_samples)
    H0_median = np.median(H0_samples)
    H0_lo, H0_hi = np.percentile(H0_samples, [16, 84])

    print("\n" + "=" * 60)
    print("DERIVED H0 POSTERIOR")
    print("=" * 60)
    print(f"H0 mean    = {H0_mean:.2f} km/s/Mpc")
    print(f"H0 median  = {H0_median:.2f} km/s/Mpc")
    print(f"H0 68% CI  = [{H0_lo:.2f}, {H0_hi:.2f}] km/s/Mpc")


if __name__ == '__main__':
    run_analysis()
