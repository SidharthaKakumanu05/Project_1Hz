
import numpy as np

def apply_ohmic_coupling(V, pairs, g_gap):
    N = V.size
    I = np.zeros(N, dtype=np.float32)
    if g_gap <= 0 or pairs.size == 0:
        return I
    i = pairs[:,0]; j = pairs[:,1]
    dV = V[j] - V[i]
    I_i = g_gap * dV
    I_j = -I_i
    np.add.at(I, i, I_i)
    np.add.at(I, j, I_j)
    return I
