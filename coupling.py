#!/usr/bin/env python3
"""
Gap junction coupling for the cerebellar microcircuit simulation.

This file implements electrical coupling between neurons through gap junctions.
Gap junctions are direct electrical connections between neurons that allow
current to flow between them, helping synchronize their activity.

For a freshman undergrad: This is like connecting two batteries with a wire -
current flows from the higher voltage to the lower voltage, and they tend
to reach the same voltage over time!
"""

import cupy as cp

def apply_ohmic_coupling(V, pairs, g_gap):
    """
    Compute electrical coupling currents between neurons connected by gap junctions.
    
    Gap junctions are like tiny electrical resistors that connect neurons directly.
    When two neurons have different voltages, current flows between them through
    the gap junction, pulling their voltages closer together.
    
    This is particularly important for IO neurons, which are electrically coupled
    to each other and tend to fire synchronously.

    Parameters
    ----------
    V : array
        Membrane voltages of all neurons in the population
    pairs : array of shape (n_pairs, 2)
        Each row is a pair of indices (i, j) saying "neuron i is electrically
        coupled to neuron j". For example, [[0,1], [1,2]] means neuron 0 is
        coupled to neuron 1, and neuron 1 is coupled to neuron 2.
    g_gap : float
        Gap-junction conductance (how strong the electrical coupling is).
        Higher values = stronger coupling = more current flows between neurons.

    Returns
    -------
    I : array
        Current to be applied to each neuron, caused by the coupling.
        Positive current = excitatory (pulls voltage up), negative = inhibitory (pulls down).
    """
    # Start with no coupling current for anyone
    I = cp.zeros_like(V, dtype=cp.float32)

    # Loop through all gap-junction pairs
    for i, j in pairs:
        dV = V[j] - V[i]          # voltage difference between the two neurons
        # Current flows from higher voltage to lower voltage
        I[i] += g_gap * dV        # neuron i feels a current pulling toward V[j]
        I[j] -= g_gap * dV        # neuron j feels equal/opposite current (Newton's 3rd law!)

    # Return total coupling currents for all neurons
    return I