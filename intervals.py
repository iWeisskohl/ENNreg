import numpy as np
from scipy.stats import norm
from Belint_ import *


def intervals(pred, level=0.9, yt=None):
    def coverage(INT, y):
        return np.mean((y >= INT[:, 0]) & (y <= INT[:, 1]))
    p = level
    mut = pred['mux']
    sigt = np.sqrt(pred['sig2x'])
    ht = pred['hx']
    nt = len(mut)
    L = mut + sigt * norm.ppf((1 - p) / 2)
    U = mut - sigt * norm.ppf((1 - p) / 2)
    INTP = np.column_stack((L, U))
    INTBel = np.zeros((nt, 2))

    for i in range(nt):
        INTBel[i, :] = belint({"mu": mut[i], "sig": sigt[i], "h": ht[i]},p)

    if yt is not None:
        coverage_P = coverage(INTP, yt)
        coverage_Bel = coverage(INTBel, yt)
    else:
        coverage_P = None
        coverage_Bel = None

    return {"INTP": INTP, "INTBel": INTBel, "coverage_P": coverage_P, "coverage_Bel": coverage_Bel}