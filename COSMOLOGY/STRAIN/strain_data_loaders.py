"""
Data loading module for Strain MCMC Inference Pipeline
Handles BAO, Pantheon+, and DES-SN5YR data loading
"""

import os
import gzip
import numpy as np
import pandas as pd
from strain_utils import make_spd_cholesky

# ============================================================================
# BAO Gaussian loader
# ============================================================================
def load_bao_gaussian(mean_path, cov_path, zmin, zmax, include_lyalpha=False):
    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"BAO mean file not found: {mean_path}")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(f"BAO covariance file not found: {cov_path}")
    rows = []
    with open(mean_path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            z = float(parts[0]); val = float(parts[1]); qty = parts[2].strip()
            if   qty == 'DM_over_rs': kind = 'DM_rd'
            elif qty == 'DH_over_rs': kind = 'DH_rd'
            elif qty == 'DV_over_rs': kind = 'DV_rd'
            else: raise ValueError(f"Unrecognized BAO quantity: {qty}")
            rows.append((z, val, kind))
    C_full = np.loadtxt(cov_path)
    if C_full.ndim != 2 or C_full.shape[0] != C_full.shape[1]:
        raise ValueError(f"BAO covariance not square: shape={C_full.shape}")
    if len(rows) != C_full.shape[0]:
        raise ValueError(f"Mean length ({len(rows)}) != covariance size ({C_full.shape[0]})")

    def is_lyalpha(z): return z >= 2.0
    mask = []
    for (z, val, kind) in rows:
        ok = (z >= zmin) and (z <= zmax)
        if ok and (not include_lyalpha) and is_lyalpha(z):
            ok = False
        mask.append(ok)
    mask = np.array(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("BAO z-window/Lyα mask removed all entries")
    sel_rows = [rows[i] for i in np.where(mask)[0]]
    y = np.array([v for (_, v, _) in sel_rows], dtype=np.float64)
    idx = np.where(mask)[0]
    C = C_full[np.ix_(idx, idx)]
    C = 0.5*(C + C.T)
    choC = make_spd_cholesky(C)
    return sel_rows, y, C, choC

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

# ============================================================================
# SN loaders
# ============================================================================
def load_pantheon_plus(data_file, cov_file, zmin, zmax):
    df = pd.read_csv(data_file, sep=r"\s+")

    mask = (
        (df["zHD"] > 0.01) &
        (df["zHD"] >= zmin) &
        (df["zHD"] <= zmax)
    )

    zHD_pantheon   = np.asarray(df["zHD"][mask], dtype=float)
    zHEL_pantheon  = np.asarray(df["zHEL"][mask], dtype=float)
    m_obs_pantheon = np.asarray(df["m_b_corr"][mask], dtype=float)

    with open(cov_file, "r") as f:
        _ = int(f.readline())
        cov_flat = np.loadtxt(f)

    C_full = cov_flat.reshape(len(df), len(df))
    idx = np.where(mask)[0]
    C = C_full[np.ix_(idx, idx)]
    C = 0.5 * (C + C.T)

    choC = make_spd_cholesky(C)

    return zHD_pantheon, zHEL_pantheon, m_obs_pantheon, C, choC

def _read_dense_cov(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Covariance file not found: {path}")
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            N = int(f.readline().strip())
            arr = np.loadtxt(f)
    else:
        with open(path, 'r') as f:
            N = int(f.readline().strip())
        arr = np.loadtxt(path, skiprows=1)
    flat = np.atleast_1d(arr).ravel()
    if flat.size < N*N:
        raise ValueError(f"DES-SN5YR cov entries ({flat.size}) < N^2 ({N*N})")
    C = flat[:N*N].reshape((N, N)).astype(np.float64)
    return 0.5*(C + C.T)

def load_des_sn_hd_and_cov(
    hd_path,
    syscov_path,
    zmin,
    zmax,
    use_syscov=True
):
    if not os.path.exists(hd_path):
        raise FileNotFoundError(f"DES-SN5YR HD CSV not found at {hd_path}")

    hd = pd.read_csv(hd_path)

    required = {'MU', 'MUERR_FINAL', 'zHD', 'zHEL'}
    if not required.issubset(hd.columns):
        raise KeyError(f"DES-SN5YR_HD.csv missing columns: {required}")

    mask = (
        (hd["zHD"] > 0.00) &
        (hd["zHD"] >= zmin) &
        (hd["zHD"] <= zmax)
    )

    if not np.any(mask):
        raise ValueError("DES-SN5YR z-cut removed all SNe")

    zHD_des5y = np.asarray(hd.loc[mask, 'zHD'], dtype=np.float64)
    zHEL_des5y = np.asarray(hd.loc[mask, 'zHEL'], dtype=np.float64)
    mu_des5y   = np.asarray(hd.loc[mask, 'MU'], dtype=np.float64)
    sig_des5y  = np.asarray(hd.loc[mask, 'MUERR_FINAL'], dtype=np.float64)

    C_stat = np.diag(sig_des5y * sig_des5y)
    C = C_stat

    if use_syscov and syscov_path and os.path.exists(syscov_path):
        C_sys_full = _read_dense_cov(syscov_path)

        if C_sys_full.shape[0] != len(hd):
            raise ValueError("DES syscov dimension mismatch with HD file")

        idx = np.where(mask)[0]
        C_sys = C_sys_full[np.ix_(idx, idx)]

        C = C_stat + C_sys

    C = 0.5 * (C + C.T)

    return zHD_des5y, zHEL_des5y, mu_des5y, C
