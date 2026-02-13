"""
Configuration module for SBGEFT Black Hole Shadow MCMC Analysis
Physical constants and SBGEFT parameters
"""

import math

# ============================================================
# Physical constants
# ============================================================
G = 6.67430e-11
c = 299792458.0
M_sun = 1.98847e30
pc = 3.085677581491367e16
rad_to_uas = 206265e6



# ============================================================
# SBGEFT constants
# ============================================================
c_log = 1.5
b0_dim = 3.0 * math.sqrt(3.0)



# ============================================================
# Source parameters
# ============================================================
M87_M = 6.5e9 * M_sun
M87_D = 16.85e6 * pc
M87_obs, M87_sig = 42.0, 3.0
a_M87 = 0.94


SGR_M = 4.297e6 * M_sun
SGR_D = 8.275e3 * pc
SGR_obs, SGR_sig = 51.8, 2.3
a_SGR = 0.1
