import numpy as np

import foncgrad_RFS

def rmsprop(psi0, verbose, options, X, y, K, eps, lambd, xi, rho, nu, optimProto, opt_rmsprop):
    go_on = True
    N = len(psi0)
    n = X.shape[0]
    r = np.zeros(N)
    error_best = np.inf
    Error = []
    t = 0
    nbatch = round(n / opt_rmsprop['batch_size'])
    psi = np.copy(psi0)

    while go_on:  # MAIN LOOP
        t += 1
        batch = np.random.choice(nbatch, n, replace=True) + 1
        error = 0

        for k in range(1, nbatch + 1):  # Loop on mini-batches
            ii = np.where(batch == k)[0]
            fg = foncgrad_RFS(psi, X[ii, :], y[ii], K, eps, lambd, xi, rho, nu, optimProto)
            error += fg['fun']

            # RMSprop
            r = opt_rmsprop['rho'] * r + (1 - opt_rmsprop['rho']) * fg['grad'] * fg['grad']
            psi = psi - (opt_rmsprop['epsi'] / (opt_rmsprop['delta'] + np.sqrt(r))) * fg['grad']

        error /= nbatch

        if error < error_best:
            psi_best = np.copy(psi)
            t_best = t
            error_best = error

        if t > options['maxiter'] or (t - t_best > opt_rmsprop['Dtmax']):
            go_on = False

        Error.append(error)

        if verbose and (t - 1) % options['print'] == 0:
            print("iter =", t, 'loss =', error)

    return {"par": psi_best, "Error": Error, "value": error_best}


