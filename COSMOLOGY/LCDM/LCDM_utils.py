"""
Utility functions module
Contains cache implementations, matrix algebra utilities, and helper functions
"""

import numpy as np
from collections import OrderedDict
from scipy.linalg import cho_factor, cho_solve


# ============================================================================
# OPTIMIZATION: LRU Cache for backgrounds and CAMB results
# ============================================================================
class LRUCache:
    def __init__(self, maxsize=50):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


_BG_CACHE = LRUCache(maxsize=60)
_CAMB_CACHE = LRUCache(maxsize=40)


# ============================================================================
# SPD jittered Cholesky
# ============================================================================
def make_spd_cholesky(C, max_tries=8):
    C = np.atleast_2d(np.array(C, dtype=float))
    n = C.shape[0]
    I = np.eye(n)
    base = 1e-12 * np.linalg.norm(C, ord=np.inf)
    jitters = [0.0] + [base * (10.0**k) for k in range(max_tries-1)]
    last_err = None
    for eps in jitters:
        try:
            return cho_factor(C + eps*I, overwrite_a=False, check_finite=False)
        except Exception as e:
            last_err = e
    raise np.linalg.LinAlgError(f"Cholesky failed after jitter escalations: {last_err}")


def quadform_with_retry(Delta, chofac, C_orig):
    try:
        x = cho_solve(chofac, Delta, check_finite=False)
        return float(Delta @ x)
    except Exception:
        chofac2 = make_spd_cholesky(C_orig)
        x = cho_solve(chofac2, Delta, check_finite=False)
        return float(Delta @ x)


def reset_caches():
    """Clear all LRU caches"""
    _BG_CACHE.clear()
    _CAMB_CACHE.clear()


def _as_int(val):
    """Safe integer conversion"""
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
