"""
Background evolution module for Strain MCMC Inference Pipeline
Implements inhomogeneity energy evolution and distance calculations
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from strain_config import c_kms, CFG
from strain_utils import _BG_CACHE

# ============================================================================ 
#  Background with inhomogeneity energy (growth–decoupled)  
# ============================================================================

def build_background_with_inhomogeneity_energy(
    H0, Om_m_eff, Om_r, Omega_inh0,
    zmax=2.8, Nz=4096, mix=0.5, max_iter=8, tol=1e-5
):
    zgrid = np.linspace(0.0, zmax, Nz)
    agrid = 1.0 / (1.0 + zgrid)

    Om_m_ini = Om_m_eff + Omega_inh0
    Om_L = 1.0 - Om_m_ini - Om_r

    S = np.ones_like(zgrid, dtype=np.float64)
    D = np.ones_like(agrid, dtype=np.float64)

    for _ in range(max_iter):

        # ===== FULL BACKGROUND =====
        E2_full = (
            Om_m_eff * (1.0 + zgrid)**3
            + Om_r * (1.0 + zgrid)**4
            + Om_L
            + Omega_inh0 * S
        )

        if np.any(~np.isfinite(E2_full)) or np.any(E2_full <= 0):
            return None, None, None

        E_full = np.sqrt(E2_full)

        # ===== GROWTH GRID =====
        a_ini = max(1.0 / (1.0 + zmax), 1e-3)
        a_asc = np.linspace(a_ini, 1.0, Nz)
        ln_a_asc = np.log(a_asc)

        # Growth background (decoupled)
        Om_LH = 1.0 - Om_m_ini - Om_r
        E2_growth = (
            Om_m_ini * a_asc**(-3)
            + Om_r * a_asc**(-4)
            + Om_LH
        )

        if np.any(~np.isfinite(E2_growth)) or np.any(E2_growth <= 0):
            return None, None, None

        E_growth = np.sqrt(E2_growth)

        dE_da_growth = np.gradient(E_growth, a_asc, edge_order=2)
        dlnH_dlna_growth = (a_asc / E_growth) * dE_da_growth
        Omega_m_a = Om_m_ini * a_asc**(-3) / E2_growth

        def ode_forward(ln_a_var, y):
            a = np.exp(ln_a_var)
            dlnH = np.interp(a, a_asc, dlnH_dlna_growth)
            Omma = np.interp(a, a_asc, Omega_m_a)

            Dv, dD_dlna = y
            rhs2 = -(dlnH + 2.0) * dD_dlna + 1.5 * Omma * Dv
            return [dD_dlna, rhs2]

        y0 = [a_ini, 1.0]
        sol = solve_ivp(
            ode_forward,
            (ln_a_asc[0], ln_a_asc[-1]),
            y0,
            t_eval=ln_a_asc,
            rtol=1e-6,
            atol=1e-8
        )

        if not sol.success or np.any(~np.isfinite(sol.y)):
            return None, None, None

        D_asc = sol.y[0].astype(np.float64)
        D_asc /= D_asc[-1]
        D = np.interp(agrid, a_asc, D_asc)

        # ===== CORRECTED SCALING =====
        H_over_H0_sq = E_full**2

        H_over_H0_sq_a = np.interp(
            agrid,
            1.0 / (1.0 + zgrid),
            H_over_H0_sq
        )

        S_new = (D**2 / agrid) * (1.0 / H_over_H0_sq_a)

        if np.any(~np.isfinite(S_new)) or S_new[0] == 0:
            return None, None, None

        S_new /= S_new[0]

        delta = np.max(np.abs(S_new - S))
        S = (1.0 - mix) * S + mix * S_new

        if delta < tol:
            break

    # FINAL BACKGROUND
    E2_final = (
        Om_m_eff * (1.0 + zgrid)**3
        + Om_r * (1.0 + zgrid)**4
        + Om_L
        + Omega_inh0 * S
    )

    if np.any(~np.isfinite(E2_final)) or np.any(E2_final <= 0):
        return None, None, None

    E_final = np.sqrt(E2_final)

    chi_grid = cumulative_trapezoid(
        1.0 / E_final, zgrid, initial=0.0
    ) * (c_kms / H0)

    return {
        'z': zgrid,
        'E': E_final,
        'chi': chi_grid,
        'S': S,
        'D': D
    }, agrid, D


# ============================================================================
# Distance engine
# ============================================================================

def build_distance_engine_dynamic(
    H0, Om_m_eff, Om_r, Omega_inh0,
    zmax=2.8, Nz=4096
):
    bg, _, _ = build_background_with_inhomogeneity_energy(
        H0, Om_m_eff, Om_r, Omega_inh0,
        zmax=zmax, Nz=Nz
    )

    if bg is None:
        return None, None, None

    zgrid = bg['z']
    Ez = bg['E']
    chi_grid = bg['chi']

    def DM(z): return np.interp(z, zgrid, chi_grid)
    def DH(z): return c_kms / (H0 * np.interp(z, zgrid, Ez))
    def DL(z): return (1.0 + z) * DM(z)

    return DM, DH, DL

# ============================================================================
# OPTIMIZATION: Cached background builder
# ============================================================================
def build_distance_engine_dynamic_cached(H0, Om_m_eff, Om_r, Omega_inh0,
                                         zmax=2.8, Nz=4096):

    cache_key = (float(H0), float(Om_m_eff), float(Om_r), float(Omega_inh0), float(zmax), int(Nz))

    if cache_key in _BG_CACHE:
        return _BG_CACHE[cache_key]

    DM, DH, DL = build_distance_engine_dynamic(H0, Om_m_eff, Om_r, Omega_inh0,
                                                zmax=zmax, Nz=Nz)

    if DM is not None:
        _BG_CACHE[cache_key] = (DM, DH, DL)

    return DM, DH, DL
