import numpy as np
from ENNreg_init import *
from ENNreg_ import *
from predict_ENNreg import *


#' Hyperparameter tuning for the ENNreg model using cross-validation
#'
#' \code{ENNreg_cv} tunes parameters xi and rho of the ENNreg model using cross-validation.
#'
#' Either the folds (a vector of the same length as y, such that \code{folds[i]} equals the
#' fold, between 1 and Kfold, containing observation i), or the number of folds must be provided.
#' Arguments \code{options} and \code{opt.rmsprop} are passed to function \code{\link{ENNreg}}.
#'
#' @param X Input matrix of size n x p, where n is the number of objects and p the number of attributes.
#' @param y Vector of length n containing observations of the response variable.
#' @param K Number of prototypes.
#' @param batch If TRUE (default), batch learning is used; otherwise, online learning is
#' used.
#' @param folds Vector of length n containing the folds (integers between 1 and Kfold).
#' @param Kfold Number of folds (default=5, used only if \code{folds} is not provided).
#' @param XI Vector of candidate values for hyperparameter \code{xi}.
#' @param RHO Vector of candidate values for hyperparameter \code{rho}.
#' @param nstart Number of random starts of the k-means algorithm (default: 100).
#' @param c Multiplicative coefficient applied to scale parameter gamma (defaut: 1).
#' @param lambda Parameter of the loss function (default=0.9).
#' @param eps Parameter of the loss function (if NULL, fixed to 0.01 times the standard
#' deviation of y).
#' @param nu Parameter of the loss function to avoid a division par zero (default=1e-16).
#' @param optimProto If TRUE (default), the initial prototypes are optimized.
#' @param verbose If TRUE (default) intermediate results are displayed.
#' @param options Parameters of the optimization algorithm (see \code{\link{ENNreg}}).
#' @param opt.rmsprop Parameters of the RMSprop algorithm (see \code{\link{ENNreg}}).
#'
#' @return A list with three components:
#' \describe{
#' \item{xi}{Optimal value of xi.}
#' \item{rho}{Optimal value of rho.}
#' \item{RMS}{Matrix of root mean squared error values}.
#' }
#' @export
#' @importFrom stats sd predict
def ENNreg_cv(X, y, K, XI, RHO, batch=True, folds=None, Kfold=5, nstart=100, c=1,
              lambda_=0.9, eps=None, nu=1e-16, optimProto=True, verbose=True,
              options={'maxiter': 1000, 'rel_error': 1e-4, 'print': 10},
              opt_rmsprop={'batch_size': 100, 'epsi': 0.001, 'rho': 0.9, 'delta': 1e-8, 'Dtmax': 100}):
    if eps is None:
        eps = 0.01 * np.std(y)

    n = len(y)

    if folds is None:
        ii = np.random.permutation(n)
        folds = np.zeros(n, dtype=int)
        for k in range(Kfold):
            folds[ii[k::Kfold]] = k + 1
    else:
        Kfold = np.max(folds)

    N1 = len(XI)
    N2 = len(RHO)
    ERRcv = np.zeros((N1, N2))
    X = np.array(X)

    for k in range(1, Kfold + 1):
        if verbose:
            print("Fold", k)

        init = ENNreg_init(X[folds != k], y[folds != k], K, nstart, c)

        for i in range(N1):
            for j in range(N2):
                fit = ENNreg(X[folds != k], y[folds != k], init=init, K=K, batch=batch,
                             lambd=lambda_, xi=XI[i], rho=RHO[j], eps=eps, nu=nu,
                             optimProto=optimProto, verbose=False, options=options, opt_rmsprop=opt_rmsprop)

                pred = predict_ENNreg(fit, newdata=X[folds == k])

                ERRcv[i, j] += np.sum((y[folds == k] - pred['mux']) ** 2)

    RMS = np.sqrt(ERRcv / n)
    imin, jmin = np.unravel_index(np.argmin(RMS), RMS.shape)

    if verbose:
        print("Best hyperparameter values:")
        print("xi =", XI[imin], "rho =", RHO[jmin])

    return {'xi': XI[imin], 'rho': RHO[jmin], 'RMS': RMS}

