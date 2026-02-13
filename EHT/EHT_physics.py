"""
Core physics functions for SBGEFT Black Hole Shadow MCMC Analysis
Strong-field metric functions and photon sphere calculations
"""

import math
import numpy as np


def A0(x):
    return 1 - 2 / x


def UEFT(x, c4):
    return 1.5 * c4 * x ** (-3) * math.log(x)


def Adef(x, c4):
    return A0(x) * (1 - 2 * UEFT(x, c4))


def dA_dx(x, c4):
    A0p = 2 / x ** 2
    U = UEFT(x, c4)
    Up = 1.5 * c4 * x ** (-4) * (1 - 3 * math.log(x))
    return A0p * (1 - 2 * U) - 2 * A0(x) * Up


def rphoton(c4):
    def F(x):
        return x * dA_dx(x, c4) - 2 * Adef(x, c4)


    a, b = 2.0, 4.0
    fa, fb = F(a), F(b)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = F(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if abs(b - a) < 1e-12:
            break
    return 0.5 * (a + b)


def bcrit(c4):
    xph = rphoton(c4)
    A = Adef(xph, c4)
    if (not np.isfinite(A)) or A <= 0:
        return np.nan, xph
    return xph / math.sqrt(A), xph
