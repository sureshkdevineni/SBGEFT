"""
SBGEFT Black Hole Shadow Analysis
Reproduces all numerical constraints in the paper.

Run:
    python EHT_main.py

Outputs:
    - validation diagnostics
    - GR baselines
    - χ² fits
    - perturbativity bounds
    - MCMC posteriors
"""

import numpy as np
from EHT_config import (
    M87_M, M87_D, M87_obs, M87_sig, a_M87,
    SGR_M, SGR_D, SGR_obs, SGR_sig, a_SGR
)
from EHT_diagnostics import static_tail_identity_check, strong_field_validation, perturbativity_report
from EHT_helpers import theta_schwarzschild, theta_kerr
from EHT_fitting import chi2_curve, summarize, c4_grid
from EHT_inference import run_mcmc_schwarzschild, run_mcmc_kerr


def run_analysis():
    """
    Main analysis workflow
    """
    
    # ============================================================
    # (1) Static tail identity check
    # ============================================================
    static_tail_identity_check()
    
    
    # ============================================================
    # (2) Strong-field validation
    # ============================================================
    strong_field_validation()
    
    
    # ============================================================
    # GR baselines
    # ============================================================
    theta_M87_S = theta_schwarzschild(M87_M, M87_D)
    theta_SGR_S = theta_schwarzschild(SGR_M, SGR_D)
    
    
    theta_M87_K = theta_kerr(M87_M, M87_D, a_M87)
    theta_SGR_K = theta_kerr(SGR_M, SGR_D, a_SGR)
    
    
    print("=== GR BASELINES ===")
    print(f"M87*:  θ_Schwarzschild = {theta_M87_S:.3f} μas")
    print(f"M87*:  θ_Kerr          = {theta_M87_K:.3f} μas")
    print(f"SgrA*: θ_Schwarzschild = {theta_SGR_S:.3f} μas")
    print(f"SgrA*: θ_Kerr          = {theta_SGR_K:.3f} μas\n")
    
    
    # ============================================================
    # TABLE 1: Schwarzschild
    # ============================================================
    print("=== TABLE 1: Schwarzschild ===")
    print(f"M87*:  θ_GR = {theta_M87_S:.3f} μas")
    print(f"SgrA*: θ_GR = {theta_SGR_S:.3f} μas\n")
    
    
    chi2_M87_S = chi2_curve(theta_M87_S, M87_obs, M87_sig)
    chi2_SGR_S = chi2_curve(theta_SGR_S, SGR_obs, SGR_sig)
    chi2_joint_S = chi2_M87_S + chi2_SGR_S
    
    
    summarize("M87*", chi2_M87_S)
    summarize("SgrA*", chi2_SGR_S)
    summarize("Joint", chi2_joint_S)
    perturbativity_report(c4_grid, chi2_joint_S, "Schwarzschild joint")

    
    
    # ============================================================
    # TABLE 2: Kerr
    # ============================================================
    print("=== TABLE 2: Kerr-corrected ===")
    print(f"M87*:  θ_GR = {theta_M87_K:.3f} μas")
    print(f"SgrA*: θ_GR = {theta_SGR_K:.3f} μas\n")
    
    
    chi2_M87_K = chi2_curve(theta_M87_K, M87_obs, M87_sig)
    chi2_SGR_K = chi2_curve(theta_SGR_K, SGR_obs, SGR_sig)
    chi2_joint_K = chi2_M87_K + chi2_SGR_K
    
    
    summarize("M87*", chi2_M87_K)
    summarize("SgrA*", chi2_SGR_K)
    summarize("Joint", chi2_joint_K)
    perturbativity_report(c4_grid, chi2_joint_K, "Kerr joint")


    
    
    # ============================================================
    # MCMC POSTERIOR: Schwarzschild (Joint)
    # ============================================================
    run_mcmc_schwarzschild(theta_M87_S, theta_SGR_S)
    
    
    # ============================================================
    # MCMC POSTERIOR: Kerr (Joint)
    # ============================================================
    run_mcmc_kerr(theta_M87_K, theta_SGR_K)
    
    
    print("All diagnostics complete.")


if __name__ == '__main__':
    run_analysis()
