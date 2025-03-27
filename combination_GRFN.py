import numpy as np

def combination_GRFN(GRFN1, GRFN2, soft=True):
    if GRFN1['h'] == 0 and GRFN2['h'] == 0:
        M = 0
        S = 1
        kappa = 0
    else:
        if soft:
            hbar = (GRFN1['h'] * GRFN2['h']) / (GRFN1['h'] + GRFN2['h'])
            rho = (hbar * GRFN1['sig'] * GRFN2['sig']) / np.sqrt((1 + hbar * GRFN1['sig']**2) * (1 + hbar * GRFN2['sig']**2))
            D = 1 + hbar * (GRFN1['sig']**2 + GRFN2['sig']**2)
            S1 = GRFN1['sig']**2 * (1 + hbar * GRFN2['sig']**2) / D
            S2 = GRFN2['sig']**2 * (1 + hbar * GRFN1['sig']**2) / D
            M1 = (GRFN1['mu'] * (1 + hbar * GRFN2['sig']**2) + GRFN2['mu'] * hbar * GRFN1['sig']**2) / D
            M2 = (GRFN2['mu'] * (1 + hbar * GRFN1['sig']**2) + GRFN1['mu'] * hbar * GRFN2['sig']**2) / D
            SIG = np.array([[S1, rho * np.sqrt(S1 * S2)], [rho * np.sqrt(S1 * S2), S2]])
        else:
            M1 = GRFN1['mu']
            M2 = GRFN2['mu']
            SIG = np.diag([GRFN1['sig']**2, GRFN2['sig']**2])

        u = np.array([GRFN1['h'], GRFN2['h']]) / (GRFN1['h'] + GRFN2['h'])
        M = np.dot(u, np.array([M1, M2]))
        S = np.sqrt(np.dot(np.dot(u, SIG), u))

        # degree of conflict
        if soft:
            if GRFN1['sig'] > 0 and GRFN2['sig'] > 0:
                kappa = 1 - np.sqrt(S1 * S2) / (GRFN1['sig'] * GRFN2['sig']) * np.sqrt(1 - rho**2) * np.exp(
                    -0.5 * (GRFN1['mu']**2 / GRFN1['sig']**2 + GRFN2['mu']**2 / GRFN2['sig']**2) +
                    0.5 / (1 - rho**2) * (M1**2 / S1 + M2**2 / S2 - 2 * rho * M1 * M2 / np.sqrt(S1 * S2)))
            elif GRFN1['sig'] > 0 and GRFN2['sig'] == 0:
                M1 = (GRFN1['mu'] + GRFN2['mu'] * hbar * GRFN1['sig']**2) / D
                kappa = 1 - 1 / np.sqrt(1 + hbar * S1) * np.exp(-0.5 * hbar / (1 + hbar * S1) * (M1 - GRFN2['mu'])**2)
            elif GRFN1['sig'] == 0 and GRFN2['sig'] > 0:
                M2 = (GRFN2['mu'] + GRFN1['mu'] * hbar * GRFN2['sig']**2) / D
                kappa = 1 - 1 / np.sqrt(1 + hbar * S2) * np.exp(-0.5 * hbar / (1 + hbar * S2) * (M2 - GRFN1['mu'])**2)
            else:
                kappa = 1 - np.exp(-0.5 * hbar * (GRFN1['mu'] - GRFN2['mu'])**2)
        else:
            kappa = 0

    return {'GRFN': {'mu': M, 'sig': S, 'h': GRFN1['h'] + GRFN2['h']}, 'conflict': kappa}

