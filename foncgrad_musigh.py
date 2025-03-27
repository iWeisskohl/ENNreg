import numpy as np
from scipy.stats import norm

def foncgrad_musigh(theta, y, eps, lambd, nu):
    n = len(y)
    mux = theta[:n]
    sig2x = theta[n:2*n]
    hx = theta[2*n:3*n]

    # Loss calculation
    sigx = np.sqrt(sig2x)
    Z2 = hx * sig2x + 1
    Z = np.sqrt(Z2)
    sig1 = sigx * Z

    A = norm.cdf(y + eps, loc=mux, scale=sigx) - norm.cdf(y - eps, loc=mux, scale=sigx)
    pl1 = 1 / Z * np.exp(-0.5 * hx * (y - eps - mux)**2 / Z2)
    pl2 = 1 / Z * np.exp(-0.5 * hx * (y + eps - mux)**2 / Z2)
    Phi1 = norm.cdf(y - eps, loc=mux, scale=sig1)
    Phi2 = norm.cdf(y + eps, loc=mux, scale=sig1)
    B = pl1 * Phi1 + pl2 * (1 - Phi2)
    PL = A + B
    Phi3 = norm.cdf(y, loc=mux, scale=sig1)
    BEL = np.maximum(0, PL - pl1 * Phi3 - pl2 * (1 - Phi3))

    ########loss function

    loss = np.mean(-lambd * np.log(BEL + nu) - (1 - lambd) * np.log(PL))

    # Gradient
    dpldh = -0.5 * Z2**(-3/2) * np.exp(-0.5 * hx * (y - eps - mux)**2 / Z2) * (sig2x + (y - eps - mux)**2 / Z2)
    dpldmu = pl1 * hx * (y - eps - mux) / Z2
    dpldsig2 = pl1 * (hx / 2) * ((hx * (y - eps - mux)**2 - Z2) / Z2**2)
    gradpl1 = np.column_stack((dpldmu, dpldsig2, dpldh))

    dpldh = -0.5 * Z2**(-3/2) * np.exp(-0.5 * hx * (y + eps - mux)**2 / Z2) * (sig2x + (y + eps - mux)**2 / Z2)
    dpldmu = pl2 * hx * (y + eps - mux) / Z2
    dpldsig2 = pl2 * (hx / 2) * ((hx * (y + eps - mux)**2 - Z2) / Z2**2)
    gradpl2 = np.column_stack((dpldmu, dpldsig2, dpldh))

    phi = norm.pdf((y - eps - mux) / sig1)
    dPhidh = -0.5 * sigx * phi * (y - eps - mux) * Z2**(-3/2)
    dPhidmu = -phi / sig1
    dPhidsig2 = -0.5 * phi * (y - eps - mux) * (sig2x * Z2)**(-3/2) * (2 * hx * sig2x + 1)
    gradPhi1 = np.column_stack((dPhidmu, dPhidsig2, dPhidh))

    phi = norm.pdf((y + eps - mux) / sig1)
    dPhidh = -0.5 * sigx * phi * (y + eps - mux) * Z2**(-3/2)
    dPhidmu = -phi / sig1
    dPhidsig2 = -0.5 * phi * (y + eps - mux) * (sig2x * Z2)**(-3/2) * (2 * hx * sig2x + 1)
    gradPhi2 = np.column_stack((dPhidmu, dPhidsig2, dPhidh))

    A = gradpl1 * Phi1.reshape(n,1) + pl1.reshape(n,1) * gradPhi1
    B = gradpl2 * (1 - Phi2.reshape(n,1)) - pl2.reshape(n,1) * gradPhi2
    phi2 = norm.pdf((y + eps - mux) / sigx)
    phi1 = norm.pdf((y - eps - mux) / sigx)
    C = -1 / sigx * (phi2 - phi1)
    D = -0.5 * sig2x**(-3/2) * (phi2 * (y + eps - mux) - phi1 * (y - eps - mux))
    gradPL = np.vstack((C, D, np.zeros_like(C))) + (A + B).T

    phi = norm.pdf((y - mux) / (sigx * Z))
    dPhidh = -0.5 * sigx * phi * (y - mux) * Z2**(-3/2)
    dPhidmu = -phi / sig1
    dPhidsig2 = -0.5 * phi * (y - mux) * (sig2x * Z2)**(-3/2) * (2 * hx * sig2x + 1)
    gradPhi3 = np.column_stack((dPhidmu, dPhidsig2, dPhidh))

    A = gradpl1 * Phi3.reshape(n,1) + pl1.reshape(n,1) * gradPhi3
    B = gradpl2 * (1 - Phi3.reshape(n,1)) - pl2.reshape(n,1) * gradPhi3
    gradBEL = gradPL - (A + B).T
    dlossdtheta = -(1 / n) * (lambd * gradBEL / np.vstack((BEL+nu,BEL+nu, BEL+nu))+ (1 - lambd) * gradPL / np.vstack((PL,PL,PL)))

    return {"fun": loss, "grad": dlossdtheta.tolist()}

