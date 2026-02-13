"""
Cosmology module for Strain MCMC Inference Pipeline
Contains CAMB integration and cosmological parameter calculations
"""

import numpy as np
import camb
from strain_config import CFG, c_kms, Tcmb_default, NEFF, MNU_EV
from strain_utils import _CAMB_CACHE

# ============================================================================
# CMB EU 3-vector & cov (DESI compressed priors θ∗, ωb, ωbc)
# ============================================================================
from strain_utils import make_spd_cholesky

CMB_EU_MEAN = np.array([0.01041027, 0.02223208, 0.14207901], dtype=np.float64)
CMB_EU_COV  = np.array([
    [6.6209942e-12,   1.24442058e-10,   -1.19287532e-09],
    [1.24442058e-10,   2.13441666e-08,   -9.40008323e-08],
    [-1.19287532e-09,   -9.40008323e-08,   1.48841714e-06]
], dtype=np.float64)
CMB_EU_COV = 0.5 * (CMB_EU_COV + CMB_EU_COV.T)
CMB_EU_CHO = make_spd_cholesky(CMB_EU_COV)

# ============================================================================
# Radiation density
# ============================================================================
def omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF):
    omega_gamma_h2 = 2.469e-5 * (Tcmb/2.7255)**4
    return omega_gamma_h2*(1+0.2271*Neff_local)/h**2

# ============================================================================
# H0 Helper
# ============================================================================
def derive_H0(Om_m_eff, ombh2, omch2, mnu_eV=MNU_EV):
    omega_nu = mnu_eV / 93.14
    h2 = (ombh2 + omch2 + omega_nu) / Om_m_eff
    if h2 <= 0:
        return None
    return 100.0 * np.sqrt(h2)

def derive_H0_E(Om_m_ini, ombh2, omch2, mnu_eV=MNU_EV):
    omega_nu = mnu_eV / 93.14
    h2 = (ombh2 + omch2 + omega_nu) / Om_m_ini
    if h2 <= 0:
        return None
    return 100.0 * np.sqrt(h2)

# ============================================================================
# OPTIMIZATION: Cached CAMB results
# ============================================================================
def get_camb_results_cached(Om_m_ini, ombh2, omch2, Neff_local=NEFF, mnu_eV=MNU_EV):

    H0 = derive_H0_E(Om_m_ini, ombh2, omch2, mnu_eV=mnu_eV)
    if H0 is None:
        return None

    cache_key = (
        float(Om_m_ini),
        float(ombh2),
        float(omch2),
        float(Neff_local),
        float(mnu_eV)
    )

    if cache_key in _CAMB_CACHE:
        return _CAMB_CACHE[cache_key]

    pars = camb.CAMBparams()

    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu_eV,
        nnu=Neff_local,
        TCMB=Tcmb_default,
        bbn_predictor=CFG['parthenope_path'],
        num_massive_neutrinos=1
    )

    pars.WantCls = False
    pars.WantTransfer = True
    pars.WantDerivedParameters = True

    try:
        results = camb.get_results(pars)
        _CAMB_CACHE[cache_key] = results
        return results
    except Exception:
        return None

# ============================================================================
# OPTIMIZATION: Cached rs_camb using CAMB cache
# ============================================================================
def rs_camb(Om_m_ini, ombh2, omch2, Neff_local=NEFF, mnu_eV=MNU_EV):
    results = get_camb_results_cached(Om_m_ini, ombh2, omch2, Neff_local, mnu_eV)
    if results is None:
        return np.nan

    try:
        drv = results.get_derived_params()
        rs = float(drv.get('rdrag', np.nan)) if isinstance(drv, dict) else float(drv.rdrag)
    except Exception:
        return np.nan

    return rs if (np.isfinite(rs) and rs > 0) else np.nan

def cmb_early_universe_vector(theta):

    Om_m_eff, ombh2, omch2, Omega_inh0 = theta
    Om_m_ini = Om_m_eff + Omega_inh0
    H0 = derive_H0_E(Om_m_ini, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None:
        return None
    omega_bc = ombh2 + omch2
    if omega_bc <= 0.0:
        return None

    mnu_cmb = float(CFG.get('cmb_mnu_eV', MNU_EV))

    results = get_camb_results_cached(
        Om_m_ini, ombh2, omch2,
        Neff_local=NEFF,
        mnu_eV=mnu_cmb
    )
    if results is None:
        return None

    try:
        drv = results.get_derived_params()
        theta_star = float(drv['thetastar']/100)
    except Exception:
        return None

    if not np.isfinite(theta_star) or theta_star <= 0.0:
        return None

    return np.array(
        [theta_star, ombh2, omega_bc],
        dtype=np.float64
    )
