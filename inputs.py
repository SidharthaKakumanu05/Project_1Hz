#!/usr/bin/env python3
"""
Input generation for the cerebellar microcircuit simulation.

This file handles the generation of external inputs to the cerebellum:
- Parallel Fibers (PF): Represent cortical input using a coin-flip model
- Mossy Fibers (MF): Represent brainstem input using Poisson processes

For a freshman undergrad: This is where we simulate the "outside world" 
sending signals into our cerebellar network!
"""

import cupy as cp

# -----------------------------
# PF state initialization
# -----------------------------
def init_pf_state(N_pool, refrac_steps):
    """
    Initialize the state dictionary for parallel fibers (PF).
    
    Parallel Fibers represent the massive input from the cerebral cortex.
    Each PF neuron needs to track its refractory state to prevent unrealistic
    high-frequency firing.

    Parameters
    ----------
    N_pool : int
        Total number of PF neurons in the pool
    refrac_steps : int
        How many time steps a PF must wait after spiking before it can spike again

    Returns
    -------
    dict
        State dictionary containing:
        - refrac: countdown timer (steps until it can fire again)
        - N: total number of PF neurons
        - refrac_steps: how many steps to stay silent after spiking
    """
    return {
        "refrac": cp.zeros(N_pool, dtype=cp.int32),  # all start ready-to-fire
        "N": N_pool,
        "refrac_steps": refrac_steps,
    }


# -----------------------------
# PF activity update (coinflip model)
# -----------------------------
def step_pf_coinflip(pf_state, rate_hz, dt):
    """
    Generate PF spikes with a coin-flip (Bernoulli) model.
    
    This function simulates how cortical input arrives at the cerebellum.
    Each PF can fire with probability p = rate * dt in each time step,
    but only if it's not in a refractory period.
    
    The coin-flip model is a simple but effective way to generate realistic
    cortical firing patterns without the computational overhead of more
    complex models.

    Parameters
    ----------
    pf_state : dict
        The PF state dictionary containing refractory timers
    rate_hz : float
        Target firing rate in Hz (e.g., 30 Hz for cortical input)
    dt : float
        Time step size in seconds

    Returns
    -------
    spikes : bool array
        Boolean array indicating which PF neurons spiked this time step
    """
    p = rate_hz * dt                  # probability of firing in this timestep
    refrac = pf_state["refrac"]       # refractory countdown for each PF
    can_fire = refrac <= 0            # only neurons with 0 refractory can spike

    # Only generate random numbers for neurons that can fire
    # This optimization saves computation when most neurons are refractory
    if cp.any(can_fire):
        randu = cp.random.random(pf_state["N"])  # uniform random numbers [0,1)
        spikes = cp.logical_and(can_fire, randu < p)  # spike if can fire AND random < probability
    else:
        spikes = cp.zeros(pf_state["N"], dtype=bool)  # no spikes if all are refractory

    # --- Update refractory state ---
    # neurons that spiked: set refractory timer (can't spike again immediately)
    refrac[spikes] = pf_state["refrac_steps"]
    # neurons that didn't spike: decrement timer if > 0 (countdown to being able to spike)
    refrac[~spikes] = cp.maximum(0, refrac[~spikes] - 1)

    pf_state["refrac"] = refrac       # save back updated refractory state
    return spikes


# -----------------------------
# Draw PF synaptic conductances
# -----------------------------
def draw_pf_conductances(N_conn, g_mean, g_std):
    """
    Sample conductance strengths for PF connections.
    
    This function generates the "strength" of each PF→PKJ connection.
    In real neurons, not all connections are equally strong - there's natural
    variability. This function captures that by drawing from a Gaussian distribution.
    
    Each connection gets a conductance value that determines how much effect
    a PF spike will have on the postsynaptic PKJ neuron.

    Parameters
    ----------
    N_conn : int
        Number of PF→PKJ connections to generate conductances for
    g_mean : float
        Mean conductance value (average strength)
    g_std : float
        Standard deviation of conductance values (how much variability)

    Returns
    -------
    g : array
        Array of conductance values for each connection
        Negative values are clipped to 0 (no negative conductance possible)
    """
    g = cp.random.normal(loc=g_mean, scale=g_std, size=N_conn).astype(cp.float32)
    return cp.maximum(g, 0.0)  # Clip negative values to 0


# -----------------------------
# Mossy fiber spikes (Poisson model)
# -----------------------------
def step_mf_poisson(N, rate_hz, dt):
    """
    Generate Poisson spikes for mossy fibers (MF).
    
    Mossy Fibers represent input from the brainstem and spinal cord.
    Unlike PFs, MFs don't need refractory periods because they're modeled
    as simple Poisson processes - each MF can fire independently in each
    time step with probability p = rate * dt.
    
    This is a simpler model than PFs because MFs are typically more
    irregular and don't show the same refractory behavior.

    Parameters
    ----------
    N : int
        Number of mossy fibers
    rate_hz : float
        Target firing rate in Hz (e.g., 50 Hz for brainstem input)
    dt : float
        Time step size in seconds

    Returns
    -------
    spikes : bool array
        Boolean array indicating which MF neurons spiked this time step
    """
    p = rate_hz * dt  # probability of firing in this timestep
    return (cp.random.random(N) < p)  # simple Poisson process