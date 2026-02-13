"""
Data setup module for Strain MCMC Inference Pipeline
Loads and prepares all observational data
"""

import numpy as np
from scipy.linalg import cho_solve
from strain_config import CFG
from strain_data_loaders import load_bao_gaussian, load_pantheon_plus, load_des_sn_hd_and_cov
from strain_utils import make_spd_cholesky
import strain_likelihoods

def setup_data():
    """
    Load and prepare all observational datasets
    """
    
    # ============================================================================
    # BAO data
    # ============================================================================
    if CFG.get('use_bao', True):
        zmin_bao, zmax_bao = CFG['zrange']['bao']
        bao_entries, y_bao, C_bao, choC_bao = load_bao_gaussian(
            CFG['bao_mean_file'], CFG['bao_cov_file'],
            zmin_bao, zmax_bao,
            include_lyalpha=CFG.get('include_lyalpha', False)
        )
        BAO_MAX_Z = max([z for (z, _, _) in bao_entries]) if bao_entries else 0.0
        
        strain_likelihoods.bao_entries = bao_entries
        strain_likelihoods.y_bao = y_bao
        strain_likelihoods.C_bao = C_bao
        strain_likelihoods.choC_bao = choC_bao
        strain_likelihoods.BAO_MAX_Z = BAO_MAX_Z

    # ============================================================================
    # Check for conflicting SN datasets
    # ============================================================================
    if CFG['use_pantheon_sn'] and CFG['use_des5y_sn']:
        raise RuntimeError(
            "Pantheon+ and DES-SN5YR cannot be used simultaneously "
            "without independent nuisance parameters."
        )

    # ============================================================================
    # Pantheon+ data
    # ============================================================================
    if CFG['use_pantheon_sn']:
        zmin_p, zmax_p = CFG['zrange']['pantheon']
        (zHD_pantheon, zHEL_pantheon, m_obs_pantheon, C_sn, choC_sn
        ) = load_pantheon_plus(
            CFG['pantheon_dat_file'],
            CFG['pantheon_cov_statsys_file'],
            zmin_p,
            zmax_p
        )

        _SN1 = np.ones_like(m_obs_pantheon)
        _SN_CINV_1 = cho_solve(choC_sn, _SN1, check_finite=False)
        _SN_CINV_1_DOT_1 = float(_SN1 @ _SN_CINV_1)
        
        strain_likelihoods.zHD_pantheon = zHD_pantheon
        strain_likelihoods.zHEL_pantheon = zHEL_pantheon
        strain_likelihoods.m_obs_pantheon = m_obs_pantheon
        strain_likelihoods.C_sn = C_sn
        strain_likelihoods.choC_sn = choC_sn
        strain_likelihoods._SN1 = _SN1
        strain_likelihoods._SN_CINV_1 = _SN_CINV_1
        strain_likelihoods._SN_CINV_1_DOT_1 = _SN_CINV_1_DOT_1

    # ============================================================================
    # DES-SN5YR data
    # ============================================================================
    if CFG.get('use_des5y_sn', False):
        zmin_d, zmax_d = CFG['zrange']['des5y']

        zHD_des5y, zHEL_des5y, mu_des5y, C_des = load_des_sn_hd_and_cov(
            CFG['des5y_hd_file'],
            CFG['des5y_syscov_file'],
            zmin=zmin_d,
            zmax=zmax_d,
            use_syscov=True
        )

        choC_des = make_spd_cholesky(C_des)
        
        strain_likelihoods.zHD_des5y = zHD_des5y
        strain_likelihoods.zHEL_des5y = zHEL_des5y
        strain_likelihoods.mu_des5y = mu_des5y
        strain_likelihoods.C_des = C_des
        strain_likelihoods.choC_des = choC_des

    # ============================================================================
    # Update zmax
    # ============================================================================
    CFG['zmax'] = max(
        CFG['zrange']['bao'][1],
        CFG['zrange']['pantheon'][1],
        CFG['zrange']['des5y'][1]
    ) + 0.1

def report_counts():
    """
    Print summary of loaded data
    """
    n_bao = int(strain_likelihoods.y_bao.size) if CFG.get('use_bao', True) else 0
    n_snP = len(strain_likelihoods.m_obs_pantheon) if CFG.get('use_pantheon_sn', False) and strain_likelihoods.m_obs_pantheon is not None else 0
    n_snD = len(strain_likelihoods.mu_des5y) if CFG.get('use_des5y_sn', False) else 0

    print(f"[DATA] BAO points used:      {n_bao}")
    print(f"[DATA] Pantheon+ SNe used:   {n_snP}")
    print(f"[DATA] DES-SN5YR SNe used:   {n_snD}")
    print(f"[DATA] CMB priors used:      {CFG.get('use_cmb_priors', False)}")
    print(f"[DATA] H0 prior used:        {CFG.get('H0_prior', {}).get('use', False)}")
    print(f"[DATA] BBN prior used:       {CFG.get('BBN_prior', {}).get('use', False)}")
