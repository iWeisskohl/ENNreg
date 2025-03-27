import numpy as np

def predict_ENNreg(object, newdata, yt=None):

    p = np.array(object['param']['W']).shape[1]

    if isinstance(newdata, (list, np.ndarray)) and p > 1:
        Xt = np.array(newdata)
    else:
        Xt = np.asarray(newdata)

    nt = Xt.shape[0]
    K = object['K']
    d = np.zeros((nt, K))
    a = np.zeros((nt, K))
    Xt = Xt.reshape(nt,p)

    for k in range(K):
        d[:, k] = np.sum((Xt - np.tile(np.array(object['param']['W'])[k, :], (nt, 1)))**2, axis=1)
        a[:, k] = np.exp(-np.array(object['param']['gam'])[k]**2 * d[:, k])

    H = np.tile(np.array(object['param']['h']), (nt, 1))
    hx = np.sum(a * H, axis=1)
    mu = np.dot(Xt.reshape(nt,1), np.array(object['param']['Beta']).T) + np.tile(np.array(object['param']['alpha']), (nt, 1))
    mux = np.sum(mu * a * H, axis=1) / hx
    sig2x = np.sum(np.tile(np.array(object['param']['sig'])**2, (nt, 1)) * a**2 * H**2, axis=1) / hx**2

    if yt is not None:
        NLL = np.mean(0.5 * np.log(2 * np.pi * sig2x) + (yt - mux)**2 / (2 * sig2x))
        RMS = np.sqrt(np.mean((yt - mux)**2))
    else:
        NLL = None
        RMS = None

    Einf = mux - np.sqrt(np.pi / (2 * hx))
    Esup = mux + np.sqrt(np.pi / (2 * hx))

    return {"mux": mux, "sig2x": sig2x, "hx": hx, "Einf": Einf, "Esup": Esup, "NLL": NLL, "RMS": RMS}