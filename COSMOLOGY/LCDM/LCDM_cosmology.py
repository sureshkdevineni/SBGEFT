"""
Cosmology module
Contains cosmological calculations, CAMB interface, and distance functions
"""

import numpy as np
import camb
from scipy.integrate import cumulative_trapezoid
from LCDM_config import c_kms, Tcmb_default, NEFF, MNU_EV, CFG
from LCDM_utils import _BG_CACHE, _CAMB_CACHE


# ============================================================================
# Radiation density
# ============================================================================
def omega_r0(h, Tcmb=Tcmb_default, Neff_local=NEFF):
    omega_gamma_h2 = 2.469e-5 * (Tcmb/2.7255)**4
    return omega_gamma_h2*(1+0.2271*Neff_local)/h**2


# ============================================================================
# H0 Helper
# ============================================================================
def derive_H0(Om_m, ombh2, omch2, mnu_eV=MNU_EV):
    omega_nu = mnu_eV / 93.14
    h2 = (ombh2 + omch2 + omega_nu) / Om_m
    if h2 <= 0:
        return None
    return 100.0 * np.sqrt(h2)


# ============================================================================
# OPTIMIZATION: Cached CAMB results
# ============================================================================
def get_camb_results_cached(Om_m, ombh2, omch2, Neff_local=NEFF, mnu_eV=MNU_EV):

    H0 = derive_H0(Om_m, ombh2, omch2, mnu_eV=mnu_eV)
    if H0 is None:
        return None

    cache_key = (
        float(Om_m),
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
def rs_camb(Om_m, ombh2, omch2, Neff_local=NEFF, mnu_eV=MNU_EV):
    results = get_camb_results_cached(Om_m, ombh2, omch2, Neff_local, mnu_eV)
    if results is None:
        return np.nan

    try:
        drv = results.get_derived_params()
        rs = float(drv.get('rdrag', np.nan)) if isinstance(drv, dict) else float(drv.rdrag)
    except Exception:
        return np.nan

    return rs if (np.isfinite(rs) and rs > 0) else np.nan


# ============================================================================
# Distances
# ============================================================================
def build_distance_engine(H0, Om_m, Om_r, zmax=2.8, Nz=4096):
    zgrid = np.linspace(0.0, zmax, Nz)
    Om_L = 1.0 - Om_m - Om_r
    E2 = Om_m*(1+zgrid)**3 + Om_r*(1+zgrid)**4 + Om_L
    if np.any(~np.isfinite(E2)) or np.any(E2 <= 0):
        return None, None, None
    E = np.sqrt(E2)
    chi_grid = cumulative_trapezoid(1.0/E, zgrid, initial=0.0) * (c_kms/H0)

    def DM(z): return np.interp(z, zgrid, chi_grid)
    def DH(z): return c_kms/(H0*np.interp(z, zgrid, E))
    def DL(z): return (1.0 + z) * DM(z)
    return DM, DH, DL


# ============================================================================
# OPTIMIZATION: Cached background builder
# ============================================================================
def build_distance_engine_cached(H0, Om_m, Om_r, zmax=2.8, Nz=4096):

    cache_key = (float(H0), float(Om_m), float(Om_r), float(zmax), int(Nz))

    if cache_key in _BG_CACHE:
        return _BG_CACHE[cache_key]

    DM, DH, DL = build_distance_engine(H0, Om_m, Om_r, zmax=zmax, Nz=Nz)

    if DM is not None:
        _BG_CACHE[cache_key] = (DM, DH, DL)

    return DM, DH, DL


# ============================================================================
# CMB early-universe vector (cached)
# ============================================================================
def cmb_early_universe_vector(theta):

    Om_m, ombh2, omch2 = theta
    H0 = derive_H0(Om_m, ombh2, omch2, mnu_eV=MNU_EV)
    if H0 is None:
        return None

    # Physical baryon + CDM density
    omega_bc = ombh2 + omch2
    if omega_bc <= 0.0:
        return None

    mnu_cmb = float(CFG.get('cmb_mnu_eV', MNU_EV))

    results = get_camb_results_cached(
        Om_m, ombh2, omch2,
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


# ============================================================================
# Derived H0 from posterior samples
# ============================================================================
def derive_h0_samples(samples, mnu_eV=MNU_EV):
    """
    Compute derived H0 samples from flat posterior samples.
    samples : array (Nsamples, 3) with columns [Om_m, ombh2, omch2]
    """
    Om_m  = samples[:, 0]
    ombh2 = samples[:, 1]
    omch2 = samples[:, 2]

    omega_nu = mnu_eV / 93.14
    h2 = (ombh2 + omch2 + omega_nu) / Om_m

    mask = h2 > 0
    H0 = np.full(len(h2), np.nan)
    H0[mask] = 100.0 * np.sqrt(h2[mask])

    return H0
