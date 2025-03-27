import numpy as np
from sklearn.cluster import KMeans


def ENNreg_init(X, y, K, nstart=100, c=1):
    X = np.array(X).reshape(-1, 1)
    p = X.shape[1]


    # Initialize cluster centers using KMeans
    clus = KMeans(n_clusters=K, max_iter=100, n_init=nstart).fit(X)
    Beta = np.zeros((K, p))
    alpha = np.zeros(K)
    sig = np.ones(K)
    W = clus.cluster_centers_
    gam = np.ones(K)

    for k in range(K):
        ii = np.where(clus.labels_ == k)[0]
        nk = len(ii)
        alpha[k] = np.mean(y[ii])

        if nk > 1:
            gam[k] = 1 / np.sqrt(clus.inertia_ / nk)
            sig[k] = np.std(y[ii])

    gam *= c
    eta = np.ones(K) * 2
    h = eta ** 2
    psi = np.concatenate((alpha, Beta.flatten(), sig, eta, gam, W.flatten()))

    init = {
        'loss': None,
        'param': {
            'alpha': alpha,
            'Beta': Beta,
            'sig': sig,
            'h': h,
            'gam': gam,
            'W': W,
            'psi': psi
        },
        'K': K,
        'pred': None,
        'Einf': None,
        'Esup': None
    }
    return init

