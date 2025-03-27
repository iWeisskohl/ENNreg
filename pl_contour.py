import numpy as np

def pl_contour(x, GRFN):
    if GRFN['h'] == 0:
        pl = np.ones(len(x))
    else:
        Z2 = GRFN['h'] * GRFN['sig']**2 + 1
        pl = 1 / np.sqrt(Z2) * np.exp(-0.5 * GRFN['h'] * (x - GRFN['mu'])**2 / Z2)
    return pl