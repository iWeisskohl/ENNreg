import numpy as np
from scipy.stats import norm
from pl_contour import *

def Bel(x, y, GRFN):
    sig1 = GRFN['sig'] * np.sqrt(GRFN['h'] * GRFN['sig']**2 + 1)
    A = norm.cdf(y, loc=GRFN['mu'], scale=GRFN['sig']) - norm.cdf(x, loc=GRFN['mu'], scale=GRFN['sig'])
    B = pl_contour(x, GRFN) * (norm.cdf((x + y) / 2, loc=GRFN['mu'], scale=sig1) - norm.cdf(x, loc=GRFN['mu'], scale=sig1))
    C = pl_contour(y, GRFN) * (norm.cdf(y, loc=GRFN['mu'], scale=sig1) - norm.cdf((x + y) / 2, loc=GRFN['mu'], scale=sig1))
    bel = np.maximum(0, A - B - C)
    return bel