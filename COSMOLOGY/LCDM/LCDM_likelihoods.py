"""
Likelihoods module
Contains chi-squared functions for BAO, Pantheon+, DES-SN5YR, and CMB priors
"""

import numpy as np
from scipy.linalg import cho_solve
from LCDM_config import CFG, NEFF, MNU_EV, Tcmb_default, CMB_EU_MEAN, CMB_EU_COV
from LCDM_cosmology import (
    derive_H0, omega_r0, rs_camb, build_distance_engine_cached,
    cmb_early_universe_vector
)
from LCDM_utils import make_spd_cholesky, quadform_with_retry


# Initialize CMB covariance Cholesky (from config)
CMB_EU_CHO = make_spd_cholesky(CMB_EU_COV)


# Global variables for BAO data (set by data_setup.py)
bao_entries = []
y_bao = np.array([], dtype=np.float64)
C_bao = None
choC_bao = None
BAO_MAX_Z = 0.0

# Global variables for Pantheon+ data (set by data_setup.py)
zHD_pantheon = np.array([])
zHEL_pantheon = np.array([])
m_obs_pantheon = np.array([])
C_sn = None
choC_sn = None
_SN1 = None
_SN_CINV_1 = None
_SN_CINV_1_DOT_1 = None

# Global variables for DES-SN5YR data (set by data_setup.py)
zHD_des5y = np.array([])
zHEL_des5y = np.array([])
mu_des5y = np.array([])
C_des = None
choC_des = None


def _bao_predict_vector(entries, DM, DH, rs):
    pred = []
    for (z, _, kind) in entries:
        if kind == 'DM_rd':
            pred.append(DM(z)/rs)
        elif kind == 'DH_rd':
            pred.append(DH(z)/rs)
        elif kind == 'DV_rd':
            dm = DM(z); dh = DH(z)
            dv = ((z*dh) * dm * dm)**(1.0/3.0)
            pred.append(dv/rs)
        else:
            raise ValueError(f"Unexpected BAO entry kind: {kind}")
    return np.array(pred, dtype=np.float64)


def bao_chi2(theta):

    if not CFG.get('use_bao', True) or y_bao.size == 0:
        return 0.0
    
    # Type guard for BAO covariance
    if C_bao is None or choC_bao is None:
        return 0.0
    
    Om_m, ombh2, omch2 = theta

    H0 = derive_H0(Om_m, ombh2, omch2)
    if H0 is None:
        return np.inf

    h = H0 / 100.0
    Om_r = omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF)

    rs = rs_camb(Om_m, ombh2, omch2)
    if not np.isfinite(rs) or rs <= 0.0:
        return np.inf

    zmax_dyn = max(CFG['zmax'], float(BAO_MAX_Z) + 0.1)
    DM, DH, _ = build_distance_engine_cached(H0, Om_m, Om_r, zmax=zmax_dyn, Nz=CFG['Nz'])

    if DM is None:
        return np.inf

    th = _bao_predict_vector(bao_entries, DM, DH, rs)
    Delta = th - y_bao
    return quadform_with_retry(Delta, choC_bao, C_bao)


def pantheon_chi2(theta):
    """
    Pantheon+ likelihood with analytic marginalization over M
    """

    if not CFG.get('use_pantheon_sn', False):
        return 0.0

    # Safe handling of potentially empty arrays
    if zHD_pantheon.size == 0 or zHEL_pantheon.size == 0:
        return 0.0
    
    # Type guards for Pantheon covariance and precomputed vectors
    if C_sn is None or choC_sn is None or _SN1 is None or _SN_CINV_1 is None or _SN_CINV_1_DOT_1 is None:
        return 0.0

    Om_m, ombh2, omch2 = theta

    H0 = derive_H0(Om_m, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None:
        return np.inf

    h = H0 / 100.0
    Om_r = omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF)

    zmax_dyn = max(float(zHD_pantheon.max()), float(zHEL_pantheon.max())) + 0.1

    DM, _, _ = build_distance_engine_cached(
        H0, Om_m, Om_r,
        zmax=zmax_dyn,
        Nz=CFG['Nz']
    )
    if DM is None:
        return np.inf

    # Angular diameter distance at zHD
    DA = DM(zHD_pantheon) / (1.0 + zHD_pantheon)

    # EXACT Pantheon definition
    mu_model = 5.0 * np.log10((1.0 + zHD_pantheon)* (1.0 + zHEL_pantheon)* DA)+ 25.0

    Delta = mu_model - m_obs_pantheon

    invC_Delta = cho_solve(choC_sn, Delta, check_finite=False)
    invC_1     = _SN_CINV_1

    a = float(Delta @ invC_Delta)
    b = float(_SN1 @ invC_Delta)
    c = _SN_CINV_1_DOT_1

    return a - b * b / c


def des5y_sn_chi2(theta):
    """
    DES-SN5YR likelihood with analytic marginalization over M
    """

    if not CFG.get('use_des5y_sn', False) or mu_des5y.size == 0:
        return 0.0

    # Safe handling of potentially empty arrays
    if zHD_des5y.size == 0:
        return 0.0
    
    # Type guard for DES covariance
    if C_des is None or choC_des is None:
        return 0.0

    Om_m, ombh2, omch2 = theta

    H0 = derive_H0(Om_m, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None:
        return np.inf

    h = H0 / 100.0
    Om_r = omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF)

    zmax_dyn = max(CFG['zmax'], float(zHD_des5y.max()) + 0.1)

    DM, _, _ = build_distance_engine_cached(
        H0, Om_m, Om_r, zmax=zmax_dyn, Nz=CFG['Nz']
    )
    if DM is None:
        return np.inf

    # Angular diameter distance at zHD
    DA = DM(zHD_des5y) / (1.0 + zHD_des5y)

    # EXACT DES definition
    mu_model = 5.0 * np.log10((1.0 + zHD_des5y) * (1.0 + zHEL_des5y) * DA) + 25.0

    Delta = mu_model - mu_des5y

    invC_Delta = cho_solve(choC_des, Delta, check_finite=False)
    invC_1     = cho_solve(choC_des, np.ones_like(mu_des5y), check_finite=False)

    a = float(Delta @ invC_Delta)
    b = float(np.ones_like(mu_des5y) @ invC_Delta)
    c = float(np.ones_like(mu_des5y) @ invC_1)

    return a - b * b / c


def cmb_chi2(theta):

    if not CFG.get('use_cmb_priors', False):
        return 0.0

    v = cmb_early_universe_vector(theta)

    if v is None or np.any(~np.isfinite(v)):
        return np.inf

    Delta = v - CMB_EU_MEAN
    return quadform_with_retry(Delta, CMB_EU_CHO, CMB_EU_COV)


def chi2_total(theta):
    chi2 = 0.0

    if CFG["use_pantheon_sn"]:
        chi2 += pantheon_chi2(theta)

    if CFG["use_des5y_sn"]:
        chi2 += des5y_sn_chi2(theta)

    if CFG["use_bao"]:
        chi2 += bao_chi2(theta)

    if CFG["use_cmb_priors"]:
        chi2 += cmb_chi2(theta)

    return chi2


def chi2_decomposition(theta):
    chi2s = {}

    chi2s["BAO"] = bao_chi2(theta) if CFG.get("use_bao", False) else 0.0

    chi2s["SN"] = 0.0
    if CFG.get("use_pantheon_sn", False):
        chi2s["SN"] += pantheon_chi2(theta)
    if CFG.get("use_des5y_sn", False):
        chi2s["SN"] += des5y_sn_chi2(theta)

    chi2s["CMB"] = cmb_chi2(theta) if CFG.get("use_cmb_priors", False) else 0.0
    chi2s["TOTAL"] = sum(chi2s.values())

    return chi2s
