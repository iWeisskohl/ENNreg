import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from ENNreg_init import *
from harris import *
from foncgrad_RFS import foncgrad_RFS
from rmsprop import *
import time




def ENNreg(X, y, init=None, K=None, batch=True, nstart=100, c=1, lambd=0.9, xi=0, rho=0,
            eps=None, nu=1e-16, optimProto=True, verbose=True,
            options={'maxiter': 1000, 'rel_error': 1e-4, 'print': 10},
            opt_rmsprop={'batch_size': 100, 'epsi': 0.001, 'rho': 0.9, 'delta': 1e-8, 'Dtmax': 100}):
    # 初始化eps
    if eps is None:
        eps = 0.01 * np.std(y)

    # 初始化K和init
    if init is None:
        if K is None:
            raise ValueError("Initial model or number of prototypes must be supplied")
        else:
            init = ENNreg_init(X, y, K, nstart, c)
    else:
        K = init['K']
    X = np.array(X).reshape(-1, 1)
    n = len(y)
    p = X.shape[1]

    if batch:
       opt_harris = [int(verbose), options['maxiter'], options['rel_error'], options['print']]
       opt = harris(foncgrad_RFS, init['param']['psi'], options=opt_harris, tr=False, X=X, y=y,
                K=K, eps=eps, lambd=lambd, xi=xi, rho=rho, nu=nu, optimProto=optimProto)
    else:
        opt = rmsprop(init['param']['psi'],verbose=verbose,options=options,X=X,y=y,K=K,eps=eps,lambd=lambd,
                 xi=xi,rho=rho,nu=nu,optimProto=optimProto,opt_rmsprop=opt_rmsprop)

    psi = opt['par']
    alpha = psi[:K]
    beta = psi[K:(K * p + K)]
    Beta = beta.reshape((K, p))
    sig = psi[(K * p + K):(K * p + 2 * K)]
    sig2 = sig**2
    eta = psi[(K * p + 2 * K):(K * p + 3 * K)]
    gam = psi[(K * p + 3 * K):(K * p + 4 * K)]
    w = psi[(K * p + 4 * K):(2 * K * p + 4 * K)]
    W = w.reshape((K, p))
    h = eta**2

    ####################### Propagation
    d = np.zeros((n, K))
    a = np.zeros((n, K))

    for k in range(K):
        d[:, k] = np.sum((X - np.tile(W[k, :], (n, 1)))**2, axis=1)
        a[:, k] = np.exp(-gam[k]**2 * d[:, k])

    H = np.tile(h, (n, 1))
    hx = np.sum(a * H, axis=1)


    mu = np.dot(X, Beta.T) + np.tile(alpha, (n, 1))
    mux = np.sum(mu * a * H, axis=1) / hx

    sig2 = np.tile(sig**2, (n, 1))
    sig2x = np.sum(sig2 * a**2 * H**2, axis=1) / hx**2


    fit = {
        'loss': opt['value'],  # You need to set this value accordingly
        'param': {
            'alpha': alpha.tolist(),
            'Beta': Beta.tolist(),
            'sig': sig.tolist(),
            'h': h.tolist(),
            'gam': gam.tolist(),
            'W': W.tolist(),
            'psi': psi.tolist()
        },
        'K': K,
        'pred': {
            's': a.tolist(),
            'mux': mux.tolist(),
            'sig2x': sig2x.tolist(),
            'hx': hx.tolist(),
            'Einf': (mux - np.sqrt(np.pi / (2 * hx))).tolist()
        },
        'Esup': (mux + np.sqrt(np.pi / (2 * hx))).tolist()
    }

    return fit