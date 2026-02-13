"""
Likelihood functions for Strain MCMC Inference Pipeline
Computes chi-squared for BAO, Pantheon+, DES-SN5YR, and CMB data
"""

import numpy as np
from scipy.linalg import cho_solve
from strain_config import CFG, NEFF, MNU_EV, Tcmb_default
from strain_cosmology import omega_r0, derive_H0, CMB_EU_MEAN, CMB_EU_CHO, CMB_EU_COV
from strain_cosmology import rs_camb, cmb_early_universe_vector
from strain_background import build_distance_engine_dynamic_cached
from strain_data_loaders import _bao_predict_vector
from strain_utils import quadform_with_retry

# Will be populated by data_setup module
bao_entries = []
y_bao = np.array([], dtype=np.float64)
C_bao = None
choC_bao = None
BAO_MAX_Z = 0.0

zHD_pantheon = np.array([])
zHEL_pantheon = np.array([])
m_obs_pantheon = np.array([])
C_sn = None
choC_sn = None
_SN1 = np.array([])
_SN_CINV_1 = np.array([])
_SN_CINV_1_DOT_1 = 0.0

zHD_des5y = np.array([])
zHEL_des5y = np.array([])
mu_des5y = np.array([])
C_des = None
choC_des = None

# ============================================================================
# Likelihoods
# ============================================================================
def bao_chi2(theta):

    if not CFG.get('use_bao', True) or y_bao.size == 0:
        return 0.0
    Om_m_eff, ombh2, omch2, Omega_inh0 = theta
    
    Om_m_ini = Om_m_eff + Omega_inh0
    
    H0 = derive_H0(Om_m_eff, ombh2, omch2, mnu_eV=MNU_EV)

    if H0 is None:
        return np.inf

    h = H0 / 100.0
    Om_r = omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF)

    rs = rs_camb(Om_m_ini, ombh2, omch2)
    if not np.isfinite(rs) or rs <= 0.0:
        return np.inf

    zmax_dyn = max(CFG['zmax'], float(BAO_MAX_Z) + 0.1)
    DM, DH, _ = build_distance_engine_dynamic_cached(H0, Om_m_eff, Om_r, Omega_inh0,
                                                     zmax=zmax_dyn, Nz=CFG['Nz'])

    if DM is None:
        return np.inf

    th = _bao_predict_vector(bao_entries, DM, DH, rs)
    Delta = th - y_bao
    return quadform_with_retry(Delta, choC_bao, C_bao)

def pantheon_chi2(theta):

    if not CFG.get('use_pantheon_sn', False):
        return 0.0
    
    # Type safety checks
    if zHD_pantheon.size == 0 or zHEL_pantheon.size == 0:
        return 0.0

    Om_m_eff, ombh2, omch2, Omega_inh0 = theta

    H0 = derive_H0(Om_m_eff, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None:
        return np.inf

    h = H0 / 100.0
    Om_r = omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF)

    zmax_dyn = max(float(np.max(zHD_pantheon)), float(np.max(zHEL_pantheon))) + 0.1

    DM, _, _ = build_distance_engine_dynamic_cached(H0, Om_m_eff, Om_r, Omega_inh0,
                                                        zmax=zmax_dyn, Nz=CFG['Nz'])
    if DM is None:
        return np.inf

    DA = DM(zHD_pantheon) / (1.0 + zHD_pantheon)

    mu_model = 5.0 * np.log10((1.0 + zHD_pantheon)* (1.0 + zHEL_pantheon)* DA)+ 25.0

    Delta = mu_model - m_obs_pantheon

    invC_Delta = cho_solve(choC_sn, Delta, check_finite=False)
    invC_1     = _SN_CINV_1

    a = float(Delta @ invC_Delta)
    b = float(_SN1 @ invC_Delta)
    c = float(_SN_CINV_1_DOT_1)

    return a - b * b / c

def des5y_sn_chi2(theta):
    """
    DES-SN5YR likelihood with analytic marginalization over M
    """

    if not CFG.get('use_des5y_sn', False) or mu_des5y.size == 0:
        return 0.0

    Om_m_eff, ombh2, omch2, Omega_inh0 = theta

    H0 = derive_H0(Om_m_eff, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None:
        return np.inf

    h = H0 / 100.0
    Om_r = omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF)

    zmax_dyn = max(CFG['zmax'], float(np.max(zHD_des5y)) + 0.1)

    DM, _, _ = build_distance_engine_dynamic_cached(H0, Om_m_eff, Om_r, Omega_inh0,
                                                        zmax=zmax_dyn, Nz=CFG['Nz'])

    if DM is None:
        return np.inf

    DA = DM(zHD_des5y) / (1.0 + zHD_des5y)

    mu_model = 5.0 * np.log10((1.0 + zHD_des5y) * (1.0 + (zHEL_des5y) * DA)) + 25.0

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
