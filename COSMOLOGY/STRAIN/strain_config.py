"""
Configuration module for Strain MCMC Inference Pipeline
Contains all configuration parameters, file paths, and initial conditions
"""

import os
from pathlib import Path
import numpy as np

# ============================================================================
# Project root & data locations
# ============================================================================

# Root = directory containing this file
ROOT = Path(__file__).resolve().parent

# Allow external override (HPC / cluster / reviewer setup)
DATA_ROOT = Path(os.environ.get("STRAIN_DATA", ROOT / "data"))

# Output directory
OUTPUT_FOLDER = "CMBPRIORSplusBAO" # user-editable label
HDF5_DIR = DATA_ROOT / "STRAIN-MCMC-CHAINS" / OUTPUT_FOLDER
HDF5_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# User-editable MCMC settings
# ============================================================================

N_WALKERS = 32     # user editable
N_BURN    = 50     # user editable
N_STEPS   = 800    # increase steps if needed until R_hat<1.01

# ============================================================================
# CONFIG
# ============================================================================

CFG = {
    'use_bao': True, # Input True or False
    'use_pantheon_sn': False, # Input True or False
    'use_des5y_sn': False, # Input True or False
    'use_cmb_priors': True, # Input True or False

    # -----------------------------
    # Input data files
    # -----------------------------
    'bao_mean_file': DATA_ROOT / 'desi_gaussian_bao_ALL_GCcomb_mean.txt',
    'bao_cov_file':  DATA_ROOT / 'desi_gaussian_bao_ALL_GCcomb_cov.txt',

    'pantheon_dat_file': DATA_ROOT / 'Pantheon+SH0ES.dat',
    'pantheon_cov_statsys_file': DATA_ROOT / 'Pantheon+SH0ES_STAT+SYS.cov',

    'des5y_hd_file': DATA_ROOT / 'DES-SN5YR_HD.csv',
    'des5y_syscov_file': DATA_ROOT / 'STAT+SYS.txt.gz',

    'parthenope_path': DATA_ROOT / 'PArthENoPE_880.2_standard.dat',

    # -----------------------------
    # Optional Gaussian priors
    # -----------------------------
    'H0_prior':  {'use': False,  'mean': 73.04,   'sigma': 1.04}, # Input True or False
    'BBN_prior': {'use': False,  'mean': 0.02218, 'sigma': 0.00055}, # Input True or False

    'Nz': 4096,
    'zmax': 2.8,
    'zrange': {'bao': (0.0, 2.5), 'pantheon': (0.0, 2.5), 'des5y': (0.0, 2.5)},
    'include_lyalpha': True,

   'mcmc': {
    'n_walkers': N_WALKERS,
    'n_burn': N_BURN,
    'n_steps': N_STEPS,
    'init_spread_frac': 0.02,
    'seed': 12345,
    'resume': True,
    },

    'cmb_mnu_eV': 0.06,
    'fixed_neff': 3.044,
    'Omega_inh_bounds': (0, 0.3),
}

# Convert Path objects → strings (for libraries expecting str)
for k, v in CFG.items():
    if isinstance(v, Path):
        CFG[k] = str(v)

# ============================================================================
# Initial parameter guesses
# ============================================================================

theta_inits = {
    1: np.array([0.31, 0.02195, 0.122, 0.01]),
    2: np.array([0.30, 0.02240, 0.120, 0.02]),
    3: np.array([0.29, 0.02285, 0.118, 0.03]),
    4: np.array([0.28, 0.02330, 0.116, 0.04]),
}

# ============================================================================
# Physical constants
# ============================================================================

c_kms = 299792.458
Tcmb_default = 2.7255
NEFF = float(CFG.get('fixed_neff', 3.044))
MNU_EV = float(CFG.get('cmb_mnu_eV', 0.06))

# ============================================================================
# CMB EU 3-vector & covariance
# ============================================================================

CMB_EU_MEAN = np.array([0.01041027, 0.02223208, 0.14207901], dtype=np.float64)

CMB_EU_COV = np.array([
    [6.6209942e-12,   1.24442058e-10,  -1.19287532e-09],
    [1.24442058e-10,  2.13441666e-08,  -9.40008323e-08],
    [-1.19287532e-09, -9.40008323e-08,  1.48841714e-06]
], dtype=np.float64)

CMB_EU_COV = 0.5 * (CMB_EU_COV + CMB_EU_COV.T)
