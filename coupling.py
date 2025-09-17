import cupy as cp


def apply_ohmic_coupling(V, pairs, g_gap):
    """
    Compute gap-junction (electrical coupling) currents for IO neurons.

    Parameters
    ----------
    V : cp.ndarray, shape [N]
        Membrane potentials of IO neurons.
    pairs : cp.ndarray, shape [M, 2]
        Each row [i, j] defines a coupled pair of IO neurons.
    g_gap : float
        Gap junction conductance.

    Returns
    -------
    I_coupling : cp.ndarray, shape [N]
        Net coupling current into each IO neuron.
    """
    N = V.size
    I_coupling = cp.zeros(N, dtype=cp.float32)

    if pairs.size == 0:
        return I_coupling

    # voltage differences per pair
    dV = V[pairs[:, 0]] - V[pairs[:, 1]]

    # current from i→j and j→i
    I_ij = -g_gap * dV
    I_ji = +g_gap * dV

    # scatter add into result
    cp.scatter_add(I_coupling, pairs[:, 0], I_ij)
    cp.scatter_add(I_coupling, pairs[:, 1], I_ji)

    return I_coupling