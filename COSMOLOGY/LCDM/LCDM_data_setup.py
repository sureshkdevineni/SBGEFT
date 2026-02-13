"""
Data setup module
Loads and prepares all observational datasets for likelihood evaluation
"""

import numpy as np
from scipy.linalg import cho_solve
from LCDM_config import CFG
from LCDM_data_loaders import (
    load_bao_gaussian, load_pantheon_plus, load_des_sn_hd_and_cov
)
from LCDM_utils import make_spd_cholesky
import LCDM_likelihoods


def setup_data():
    """
    Load all datasets and initialize global variables in likelihoods module
    """
    
    # ============================================================================
    # Load BAO data
    # ============================================================================
    if CFG.get('use_bao', True):
        zmin_bao, zmax_bao = CFG['zrange']['bao']
        bao_entries, y_bao, C_bao, choC_bao = load_bao_gaussian(
            CFG['bao_mean_file'], CFG['bao_cov_file'],
            zmin_bao, zmax_bao,
            include_lyalpha=CFG.get('include_lyalpha', False)
        )
        BAO_MAX_Z = max([z for (z, _, _) in bao_entries]) if bao_entries else 0.0
        
        # Update likelihoods module globals
        LCDM_likelihoods.bao_entries = bao_entries
        LCDM_likelihoods.y_bao = y_bao
        LCDM_likelihoods.C_bao = C_bao
        LCDM_likelihoods.choC_bao = choC_bao
        LCDM_likelihoods.BAO_MAX_Z = BAO_MAX_Z

    # ============================================================================
    # Load Pantheon+ data
    # ============================================================================
    if CFG.get('use_pantheon_sn', False):
        if CFG.get('use_des5y_sn', False):
            raise RuntimeError(
                "Pantheon+ and DES-SN5YR cannot be used simultaneously "
                "without independent nuisance parameters."
            )
        
        zmin_p, zmax_p = CFG['zrange']['pantheon']
        (zHD_pantheon, zHEL_pantheon, m_obs_pantheon, C_sn, choC_sn
        ) = load_pantheon_plus(
            CFG['pantheon_dat_file'],
            CFG['pantheon_cov_statsys_file'],
            zmin_p,
            zmax_p
        )

        # Precompute Pantheon+ analytic M-marginalization vectors
        _SN1 = np.ones_like(m_obs_pantheon)
        _SN_CINV_1 = cho_solve(choC_sn, _SN1, check_finite=False)
        _SN_CINV_1_DOT_1 = float(_SN1 @ _SN_CINV_1)
        
        # Update likelihoods module globals
        LCDM_likelihoods.zHD_pantheon = zHD_pantheon
        LCDM_likelihoods.zHEL_pantheon = zHEL_pantheon
        LCDM_likelihoods.m_obs_pantheon = m_obs_pantheon
        LCDM_likelihoods.C_sn = C_sn
        LCDM_likelihoods.choC_sn = choC_sn
        LCDM_likelihoods._SN1 = _SN1
        LCDM_likelihoods._SN_CINV_1 = _SN_CINV_1
        LCDM_likelihoods._SN_CINV_1_DOT_1 = _SN_CINV_1_DOT_1

    # ============================================================================
    # Load DES-SN5YR data
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
        
        # Update likelihoods module globals
        LCDM_likelihoods.zHD_des5y = zHD_des5y
        LCDM_likelihoods.zHEL_des5y = zHEL_des5y
        LCDM_likelihoods.mu_des5y = mu_des5y
        LCDM_likelihoods.C_des = C_des
        LCDM_likelihoods.choC_des = choC_des

    # ============================================================================
    # Update maximum redshift
    # ============================================================================
    CFG['zmax'] = max(
        CFG['zrange']['bao'][1],
        CFG['zrange']['pantheon'][1],
        CFG['zrange']['des5y'][1]
    ) + 0.1


if __name__ == '__main__':
    setup_data()
    print("Data setup complete!")
