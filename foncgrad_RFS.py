import numpy as np
from scipy.stats import norm
'''
def foncgrad_RFS(psi, X, y, K, eps, lambda_val, xi, rho, nu, optimProto):
    p = X.shape[1]
    n = len(y)

    # Parameter extraction
    alpha = psi[:K]
    beta = psi[K:K * p + K]
    Beta = beta.reshape(K, p)
    sig = psi[K * p + K:K * p + 2 * K]
    sig2 = sig**2
    eta = psi[K * p + 2 * K:K * p + 3 * K]
    gam = psi[K * p + 3 * K:K * p + 4 * K]
    w = psi[K * p + 4 * K:2 * K * p + 4 * K]
    W = w.reshape(K, p)
    h = eta**2

    # Propagation
    d = np.zeros((n, K))
    a = np.zeros((n, K))

    for k in range(K):
        d[:, k] = np.sum((X - np.tile(W[k, :], (n, 1)))**2, axis=1)
        a[:, k] = np.exp(-gam[k]**2 * d[:, k])

    H = np.tile(h, (n, 1))
    aH = a * H
    hx = np.sum(aH, axis=1)
    S2 = hx**2

    mu = np.dot(X, Beta.T) + np.tile(alpha, (n, 1))
    mux = np.sum(mu * aH, axis=1) / hx
    sig2x = np.sum(np.tile(sig2, (n, 1)) * aH**2, axis=1) / S2

    # Loss calculation
    theta = np.concatenate((mux, sig2x, hx))
    fg = foncgrad_musigh(theta, y, eps, lambda_val, nu)
    loss = fg["fun"] + xi * np.mean(h) + rho * np.mean(gam**2)
    dlossdtheta = np.array(fg["grad"]).reshape(3, n, order='F')
'''
import numpy as np

from foncgrad_musigh import *
def foncgrad_RFS(psi, X, y, K, eps, lambd, xi, rho, nu, optimProto):
    n, p = X.shape

    alpha = psi[:K]
    beta = psi[K:(K * p + K)]
    Beta = beta.reshape(K, p,order="F")
    sig = psi[(K * p + K):(K * p + 2 * K)]
    sig2 = sig ** 2

    eta = psi[(K * p +2* K):(K * p + 3 * K)]
    gam = psi[(K * p + 3 * K):(K * p + 4 * K)]
    w = psi[(K * p + 4 * K):(2 * K * p + 4 * K)]
    W = w.reshape(K, p,order="F")
    h = eta ** 2

    d = np.zeros((n, K))
    a = np.zeros((n, K))
    for k in range(K):
        d[:, k] = np.sum((X - np.tile(W[k, :], (n, 1))) ** 2, axis=1)
        a[:, k] = np.exp(-gam[k] ** 2 * d[:, k])

    H = np.tile(h, (n, 1))
    aH = a * H
    hx = np.sum(aH, axis=1)
    S2 = hx ** 2
    mu = np.dot(X, Beta.T) + np.tile(alpha, (n, 1))
    mux = np.sum(mu * aH, axis=1) / hx
    sig2x = np.sum(np.tile(sig2, (n, 1)) * aH ** 2, axis=1) / S2
    theta = np.concatenate((mux, sig2x, hx))
    fg = foncgrad_musigh(theta, y, eps, lambd, nu)
    loss = fg['fun'] + xi * np.mean(h) + rho * np.mean(gam ** 2)
    dlossdtheta = np.array(fg['grad']).reshape(3, n)

    test=np.repeat(hx.reshape(n,1),K, axis=1)
    dmuxdmu = aH / test
    dlossdbeta = np.zeros((K, p))
    for k in range(K):
        dlossdbeta[k, :]=np.dot(dlossdtheta[0,:] * dmuxdmu[:, k], X)
    dlossdalpha = np.dot(dlossdtheta[0, :], dmuxdmu)

    dsig2xdsig2 = aH ** 2 / S2.reshape(n,1)
    dlossdsig2 = np.dot(dlossdtheta[1, :], dsig2xdsig2)
    dlossdsig = 2 * sig * dlossdsig2

    dhxdh = a
    dmuxdh = a * (mu - np.tile(mux, (K, 1)).T) / np.tile(hx, (K, 1)).T
    test0=sig2 * h
    dsig2xdh = 2 * a * (a * np.tile(sig2 * h,(n,1)) - np.tile(hx * sig2x, (K, 1)).T) / S2.reshape(n,1)
    dlossdh = np.dot(dlossdtheta[2, :], dhxdh) + np.dot(dlossdtheta[0, :], dmuxdh) + np.dot(dlossdtheta[1, :],
                                                                                            dsig2xdh) + xi / K
    dlossdeta = 2 * eta * dlossdh

    dhxda = H
    dmuxda = H * (mu - np.tile(mux, (K, 1)).T) / hx.reshape(n,1)
    dsig2xda = 2 * H * (a * np.repeat(test0.reshape(1,K), n,axis=0) - np.tile(hx * sig2x, (K, 1)).T) / S2.reshape(n,1)
    dlossda = np.tile(dlossdtheta[0, :],(K,1)).T* dmuxda + np.tile(dlossdtheta[1, :],(K,1)).T*dsig2xda + np.tile(dlossdtheta[2, :],(K,1)).T* dhxda
    dadgamma = -2 * gam.reshape(1, K) * d * a
    dlossdgamma = np.sum(dlossda*dadgamma, axis=0) + 2 * rho * gam / K

    dlossdw = np.zeros((K, p))
    if optimProto:
        for k in range(K):
            dajdw = 2 * gam[k] ** 2 * (X - np.tile(W[k, :], (n, 1))) * a[:, k].reshape(n,1)
            dlossdw[k, :] =np.dot(dlossda[:, k] ,dajdw)

    grad = np.concatenate((dlossdalpha, dlossdbeta.flatten(), dlossdsig, dlossdeta, dlossdgamma, dlossdw.flatten()))

    return {'fun': loss, 'grad': grad}
