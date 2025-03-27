import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm
from Bel import *


def belint (GRFN, level=0.9):
    alpha = level
    a = -GRFN['sig'] * norm.ppf((1 - alpha) / 2)
    b = 2 * a
    while Bel(GRFN['mu'] - b, GRFN['mu'] + b, GRFN) < alpha:
        b = 2 * b

    def equation(r, GRFN, alpha):
        return Bel(GRFN['mu'] - r, GRFN['mu'] + r, GRFN) - alpha

    result = root_scalar(equation, bracket=[a, b], args=(GRFN, alpha))
    r = result.root
    return [GRFN['mu'] - r, GRFN['mu'] + r]