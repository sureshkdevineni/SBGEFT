"""
Validation diagnostics for SBGEFT Black Hole Shadow MCMC Analysis
Static tail identity check and strong-field validation
"""

import math
import numpy as np
from EHT_config import c_log, b0_dim
from EHT_physics import bcrit


def static_tail_identity_check():
    """
    (1) Static tail identity check
    """
    print("=== (1) Static tail identity ===")


    r = np.linspace(10.0, 1e4, 200000)
    h = r[1] - r[0]
    f = c_log * (1.0 / r) * np.log(r)


    fprime = np.empty_like(f)
    fprime[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    fprime[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)
    fprime[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)


    fpp = np.empty_like(f)
    fpp[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (h * h)
    fpp[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / (h * h)
    fpp[-1] = (2 * f[-1] - 5 * f[-2] + 4 * f[-3] - f[-4]) / (h * h)


    lap = fpp + (2.0 / r) * fprime


    fit_lo = int(0.2 * len(r))
    X = 1.0 / (r[fit_lo:] ** 3)
    A_fit = float(np.dot(X, lap[fit_lo:]) / np.dot(X, X))
    rel_err = abs(A_fit + c_log) / c_log


    print(f"A_fit = {A_fit:.12f} (target = -1.5)")
    print(f"Relative error = {rel_err:.3e}\n")


def strong_field_validation():
    """
    (2) Strong-field validation
    """
    print("=== (2) Strong-field validation ===")


    # Test point
    c4_test = 0.2
    bc, xph = bcrit(c4_test)


    print(f"r_ph/rg = {xph:.9f}  (GR=3)")
    print(f"b_c/rg  = {bc:.9f}  (GR=5.196)")


    db_num = (bc - b0_dim) / b0_dim
    db_ana = c4_test * math.log(3) / 18


    print(f"Δb/b numeric  = {db_num:.6e}")
    print(f"Δb/b analytic = {db_ana:.6e}")


    c4_lin = np.linspace(0, 0.3, 60)
    bc_exact = np.array([bcrit(c)[0] for c in c4_lin])
    bc_lin = b0_dim * (1 + c4_lin * math.log(3) / 18)


    delta_exact = (bc_exact - b0_dim) / b0_dim
    delta_lin   = (bc_lin   - b0_dim) / b0_dim

    # avoid division at c4 = 0
    mask = np.abs(delta_lin) > 1e-14
    rel_dev = np.zeros_like(delta_lin)
    rel_dev[mask] = np.abs((delta_exact[mask] - delta_lin[mask]) / delta_lin[mask])

    dev = np.max(rel_dev)



    print(f"Max linearity deviation: {100*dev:.3f}%\n")

def perturbativity_report(c4_grid, chi2_joint, label):
    
    """
    Compute |U_EFT(3 r_g)| at best fit and 1σ upper bound
    """
    print(f"=== Perturbativity at 3 r_g ({label}) ===")

    # Photon-sphere EFT amplitude:
    # U(3rg) = (ln 3 / 18) * c4
    def U_ps(c4):
        return (math.log(3.0) / 18.0) * c4

    valid = np.isfinite(chi2_joint)
    chi2_valid = chi2_joint[valid]
    c4_valid = c4_grid[valid]

    i = int(np.argmin(chi2_valid))
    c4_best = float(c4_valid[i])
    chi2_min = float(chi2_valid[i])

    target = chi2_min + 1.0
    c4_upper = None
    for j in range(i, len(c4_valid)):
        if chi2_valid[j] > target:
            c4_upper = float(c4_valid[j])
            break

    U_best = abs(U_ps(c4_best))
    U_hi = abs(U_ps(c4_upper if c4_upper is not None else c4_best))

    print(f"  |U_EFT(3 r_g)| best  = {U_best:.3e}")
    print(f"  |U_EFT(3 r_g)| 1σ up = {U_hi:.3e}\n")

