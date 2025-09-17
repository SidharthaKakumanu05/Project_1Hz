import cupy as cp


def apply_ohmic_coupling(V, pairs, g_gap):
    """
    Simple gap-junction coupling current.
    pairs: [n_pairs, 2] indices
    """
    I = cp.zeros_like(V, dtype=cp.float32)
    for i, j in pairs:
        dV = V[j] - V[i]
        I[i] += g_gap * dV
        I[j] -= g_gap * dV
    return I