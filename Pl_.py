import numpy as np
from scipy.stats import norm
from pl_contour import *
def Pl(x, y, GRFN):
    sig1 = GRFN['sig'] * np.sqrt(GRFN['h'] * GRFN['sig']**2 + 1)
    A = norm.cdf(y, loc=GRFN['mu'], scale=GRFN['sig']) - norm.cdf(x, loc=GRFN['mu'], scale=GRFN['sig'])
    B = pl_contour(x, GRFN) * norm.cdf(x, loc=GRFN['mu'], scale=sig1)
    C = pl_contour(y, GRFN) * (1 - norm.cdf(y, loc=GRFN['mu'], scale=sig1))
    pl = A + B + C
    return pl