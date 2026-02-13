"""
MCMC sampler module
Contains functions to initialize walkers and run ensemble MCMC sampling
"""

import os
import numpy as np
import emcee
from emcee.backends import HDFBackend
from LCDM_config import CFG, HDF5_DIR
from LCDM_priors import log_prior, log_probability
from LCDM_utils import _as_int


def initialize_walkers(theta0, n_walkers, spread, rng):
    p0 = []
    while len(p0) < n_walkers:
        trial = theta0 * (1.0 + spread * rng.normal(size=theta0.size))
        if np.isfinite(log_prior(trial)):
            p0.append(trial)
    return np.array(p0)


def set_mcmc_files(run_id):
    CFG['mcmc']['burnin_h5'] = os.path.join(
        HDF5_DIR, f"chain{run_id}_burnin.h5"
    )
    CFG['mcmc']['prod_h5'] = os.path.join(
        HDF5_DIR, f"chain{run_id}_prod.h5"
    )


def run_mcmc():
    n_dim = 3
    mcfg = CFG['mcmc']
    n_walkers = int(mcfg['n_walkers'])
    n_burn    = int(mcfg['n_burn'])
    n_steps   = int(mcfg['n_steps'])
    spread    = float(mcfg['init_spread_frac'])
    seed      = int(mcfg['seed'])
    burnin_h5 = mcfg.get('burnin_h5', None)
    prod_h5   = mcfg.get('prod_h5', None)
    resume    = bool(mcfg.get('resume', True))

    if n_walkers < 2 * n_dim:
        raise ValueError("emcee requires n_walkers >= 2 * n_dim")

    rng = np.random.default_rng(seed)
    theta0 = np.array(CFG['mcmc']['theta_init'], dtype=float)
    
    # --------------------
    # Burn-in
    # --------------------
    burn_backend = HDFBackend(burnin_h5)
    sampler_burn = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability, backend=burn_backend
    )

    iter_burn: int = _as_int(burn_backend.iteration)

    if iter_burn == 0 or not resume:
        if iter_burn > 0 and not resume:
            print(
                f"[MCMC] burn-in file {burnin_h5} exists but resume=False; resetting."
            )
        burn_backend.reset(n_walkers, n_dim)
        p0_burn = initialize_walkers(theta0, n_walkers, spread, rng)
        print(
            f"[MCMC] Starting new burn-in: {n_burn} steps, {n_walkers} walkers..."
        )
        sampler_burn.run_mcmc(p0_burn, n_burn, progress=True)

    elif iter_burn < n_burn:
        steps_done: int = iter_burn
        steps_remain: int = n_burn - steps_done
        print(
            f"[MCMC] Resuming burn-in from iteration {steps_done} of {n_burn} "
            f"in {burnin_h5} ({steps_remain} steps remaining)..."
        )
        sampler_burn.run_mcmc(None, steps_remain, progress=True)

    else:
        print(
            f"[MCMC] Burn-in already complete in {burnin_h5} "
            f"(iterations={iter_burn})."
        )

    last_state = burn_backend.get_last_sample()
    p0_prod = last_state.coords

    # --------------------
    # Production
    # --------------------
    prod_backend = HDFBackend(prod_h5)
    sampler_prod = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability, backend=prod_backend
    )

    iter_prod: int = _as_int(prod_backend.iteration)

    if iter_prod == 0 or not resume:
        if iter_prod > 0 and not resume:
            print(
                f"[MCMC] production file {prod_h5} exists but resume=False; resetting."
            )
        prod_backend.reset(n_walkers, n_dim)
        print(
            f"[MCMC] Starting new production run: {n_steps} steps, {n_walkers} walkers..."
        )
        sampler_prod.run_mcmc(p0_prod, n_steps, progress=True)

    elif iter_prod < n_steps:
        steps_done: int = iter_prod
        steps_remain: int = n_steps - steps_done
        print(
            f"[MCMC] Resuming production from iteration {steps_done} of {n_steps} "
            f"in {prod_h5} ({steps_remain} steps remaining)..."
        )
        sampler_prod.run_mcmc(None, steps_remain, progress=True)

    else:
        print(
            f"[MCMC] Production already complete in {prod_h5} "
            f"(iterations={iter_prod})."
        )

    # --------------------
    # Results
    # --------------------
    sampler = sampler_prod
    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    logP = sampler.get_log_prob(discard=0, thin=1, flat=True)

    print(
        f"[MCMC] Mean acceptance fraction = "
        f"{np.mean(sampler.acceptance_fraction):.3f}"
    )

    return sampler, flat_samples, logP
