import numpy as np
#import ENNreg
#import ENNreg_init

def ENNreg_holdout(X, y, K, XI, RHO, batch=True, val=None, nval=None, nstart=100, c=1,
                   lambda_=0.9, eps=None, nu=1e-16, optimProto=True, verbose=True,
                   options={'maxiter': 1000, 'rel.error': 1e-4, 'print': 10},
                   opt_rmsprop={'batch_size': 100, 'epsi': 0.001, 'rho': 0.9, 'delta': 1e-8, 'Dtmax': 100}):
    if eps is None:
        eps = 0.01 * np.std(y)

    n = len(y)

    if val is None:
        if nval is None:
            raise ValueError("Validation set or number of validation instances must be supplied")
        else:
            val = np.random.choice(n, nval, replace=False)
    else:
        nval = len(val)

    N1 = len(XI)
    N2 = len(RHO)
    RMS = np.zeros((N1, N2))
    X = np.array(X)

    init = ENNreg_init(X[~np.isin(np.arange(n), val)], y[~np.isin(np.arange(n), val)], K, nstart, c)

    for i in range(N1):
        for j in range(N2):
            fit = ENNreg(X[~np.isin(np.arange(n), val)], y[~np.isin(np.arange(n), val)], init=init, K=K, batch=batch,
                         lambd=lambd, xi=XI[i], rho=RHO[j], eps=eps, nu=nu,
                         optimProto=optimProto, verbose=False, options=options, opt_rmsprop=opt_rmsprop)

            pred = predict_ENNreg(fit, newdata=X[val], y=y[val])
            RMS[i, j] = pred['RMS']

            if verbose:
                print("xi =", XI[i], "rho =", RHO[j], "rms =", RMS[i, j])

    imin, jmin = np.unravel_index(np.argmin(RMS), RMS.shape)

    if verbose:
        print("Best hyperparameter values:")
        print("xi =", XI[imin], "rho =", RHO[jmin])

    return {'xi': XI[imin], 'rho': RHO[jmin], 'RMS': RMS}

