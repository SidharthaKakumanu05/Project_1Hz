import cupy as cp

def apply_ohmic_coupling(V, pairs, g_gap):
    """
    Compute electrical coupling currents between neurons
    connected by gap junctions (like little resistors between cells).

    Parameters
    ----------
    V : array
        Membrane voltages of all neurons in the population.
    pairs : array of shape (n_pairs, 2)
        Each row is a pair of indices (i, j) saying "neuron i is electrically
        coupled to neuron j".
    g_gap : float
        Gap-junction conductance (how strong the electrical coupling is).

    Returns
    -------
    I : array
        Current to be applied to each neuron, caused by the coupling.
    """
    # Start with no coupling current for anyone
    I = cp.zeros_like(V, dtype=cp.float32)

    # Loop through all gap-junction pairs
    for i, j in pairs:
        dV = V[j] - V[i]          # voltage difference between the two neurons
        I[i] += g_gap * dV        # neuron i feels a current pulling toward V[j]
        I[j] -= g_gap * dV        # neuron j feels equal/opposite current

    # Return total coupling currents for all neurons
    return I