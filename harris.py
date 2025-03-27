
import numpy as np
import time
'''
def harris(fun, x, options=[1, 100, 1e-4, 10], tr=False, *args, **kwargs):
    pas = np.full(len(x), 0.1)
    a = 1.2
    b = 0.8
    c = 0.5
    ovf = 1e4
    unf = 1e-6
    alpha = 0.9

    it = 0
    gain = 1

    fungrad = fun(x, *args, **kwargs)
    yp = fungrad['fun']
    gp = fungrad['grad']
    xp = np.copy(x)
    x = xp - pas * gp

    if tr:
        Trace = {'time': np.zeros((options[1] + 1, 3)), 'fct': np.zeros(options[1] + 1)}
        Trace['time'][0, :] = [0, 0, 0]
        Trace['fct'][0] = yp
        start_time = time.process_time()
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        ptm = np.array([start_time, end_time, elapsed_time])
    else:
        Trace = None

    while (gain / abs(yp) >= options[2]) and (it < options[1]):
        it += 1
        fungrad = fun(x, *args, **kwargs)
        y = fungrad['fun']
        g = fungrad['grad']

        if tr:
            current_time = np.array([0, 0, 0])
            current_time[:3] = np.array(np.round(np.subtract(np.array(time.time()), ptm), 3))
            Trace['time'][it, :] = current_time
            Trace['fct'][it] = y

        if options[0] > 0:
            if it % options[3] == 1:
                print([it, y, gain / abs(yp)])
'''


import numpy as np
import time
from foncgrad_RFS import foncgrad_RFS
def harris(grad_function, x, options=(1, 100, 1e-4, 10), tr=False, *args, **kwargs):
    pas = np.full(len(x), 0.1)
    a = 1.2
    b = 0.8
    c = 0.5
    ovf = 1e4
    unf = 1e-6
    alpha = 0.9

    it = 0
    gain = 1
    #grad_function=foncgrad_RFS
    fungrad = foncgrad_RFS(x, *args, **kwargs)
    yp = fungrad['fun']
    gp = fungrad['grad']
    xp = x.copy()
    x = xp - pas * gp
    # New part for trace
    if tr:
        Trace = {'time': np.zeros((options[1] + 1, 3)), 'fct': np.zeros(options[1] + 1)}
        Trace['time'][0, :] = [0, 0, 0]
        Trace['fct'][0] = yp
        ptm = time.process_time()
    else:
        Trace = None

    while (gain / abs(yp) >= options[2]) and (it < options[1]):
        it += 1
        fungrad = foncgrad_RFS(x, *args, **kwargs)
        y = fungrad['fun']
        g = fungrad['grad']

        # ---
        if tr:
            elapsed_time = time.process_time() - ptm
            Trace['time'][it, :] = [elapsed_time, elapsed_time, elapsed_time]
            Trace['fct'][it] = y
        # ---

        if options[0] > 0:
            if it % options[3] == 1:
                print([it, y, gain / abs(yp)])

        if y > yp:
            x = xp.copy()
            g = gp.copy()
            pas = pas * c
            x = x - pas * g
        else:
            gain = alpha * gain + (1 - alpha) * abs(yp - y)
            xp = x.copy()
            test = (g * gp) >= 0
            pas = (test * a + (1 - test) * b) * pas
            pas = np.where(pas <= ovf, pas, ovf)
            pas = np.where(pas >= unf, pas, unf)
            gp = g.copy()
            x = x - pas * g
            yp = y

    return {'par': x, 'value': y, 'trace': Trace}
