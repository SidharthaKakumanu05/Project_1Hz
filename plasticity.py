#!/usr/bin/env python3
"""
Synaptic plasticity rules for the cerebellar microcircuit simulation.

This file implements the learning rules for PF→PKJ synapses, which is the core
of cerebellar learning. The cerebellum learns by changing the strength of
connections between parallel fibers and Purkinje cells based on the timing
of different types of input.

For a freshman undergrad: This is where the cerebellum "learns"! When certain
patterns of activity happen, the connections between neurons get stronger or weaker.
"""

import cupy as cp

def update_pfpkj_plasticity(
    w, pre_idx, post_idx,
    pf_spikes, pkj_spikes, cf_mask,
    t, cfg,
    last_pf_spike, last_pkj_spike, last_cf_spike
):
    """
    Update synaptic weights for parallel fiber (PF) → Purkinje cell (PKJ) connections.
    
    Implements the natural LTD/LTP mechanism driven by IO neurons' 1 Hz firing rate and coupling.
    This is the core mechanism for long-term PKJ cell stability:
    
    - LTD (Long-Term Depression): PF spikes followed by CF spikes weaken connections
      This is the main learning mechanism - when a PF predicts a CF error signal,
      that connection gets weaker (the cerebellum "learns" to ignore that input)
    
    - LTP (Long-Term Potentiation): PF spikes followed by PKJ spikes strengthen connections
      This happens when PF input successfully drives PKJ firing without CF error signals
      (the cerebellum "learns" that this input is useful)

    Parameters
    ----------
    w : array
        Current synaptic weights for all PF→PKJ connections
    pre_idx, post_idx : arrays
        Indices mapping each synapse to its presynaptic PF and postsynaptic PKJ
    pf_spikes : bool array
        Which PFs spiked at this time step
    pkj_spikes : bool array
        Which PKJs spiked at this time step
    cf_mask : bool array
        Which PKJs received a climbing fiber (CF) spike this step (error signal)
    t : float
        Current simulation time (in seconds)
    cfg : dict
        Simulation parameters (plasticity windows, scales, weight limits)
    last_pf_spike, last_pkj_spike, last_cf_spike : arrays
        Record of most recent spike times for each neuron type

    Returns
    -------
    w : array
        Updated synaptic weights after applying plasticity rules
    last_pf_spike, last_pkj_spike, last_cf_spike : arrays
        Updated spike time records
    """

    # --- Plasticity configuration parameters ---
    # These parameters control how much and when learning happens
    ltd_win   = cp.float32(cfg["ltd_window"]["t_pre_cf"])   # LTD window size (e.g., 100 ms)
    ltp_win   = cp.float32(cfg["ltp_window"]["t_pre_cf"])   # LTP window size (e.g., 900 ms)
    ltd_scale = cp.float32(cfg["ltd_scale"])                # How much weight decreases for LTD
    ltp_scale = cp.float32(cfg["ltp_scale"])                # How much weight increases for LTP
    w_min     = cp.float32(cfg["w_min"])                    # Lower bound on synaptic weight
    w_max     = cp.float32(cfg["w_max"])                    # Upper bound on synaptic weight
    # No weight leak - CbmSim uses LTP/LTD equilibrium for forgetting
    # w_leak    = cp.float32(0.0)                           # CbmSim doesn't use weight leak

    # --- Update spike history for this step ---
    # We need to keep track of when each neuron last spiked to calculate timing
    if pf_spikes.any():
        last_pf_spike[pf_spikes] = t              # record time for all spiking PFs
    if pkj_spikes.any():
        last_pkj_spike[pkj_spikes] = t            # record time for all spiking PKJs
    if cf_mask.any():
        cf_idx = cp.where(cf_mask)[0]             # which PKJs got CF this step
        last_cf_spike[cf_idx] = t                 # record time for all CF spikes

    # --- Compute how long ago each PF fired (per synapse) ---
    # This tells us the timing relationship between PF spikes and current events
    dt_pf = t - last_pf_spike[pre_idx]            # time since presyn PF spiked

    # --- LTD rule: PF must precede CF within window ---
    # This is the main learning mechanism: "If PF predicts CF error, weaken that connection"
    if cf_mask.any():
        # posts_cf marks synapses whose postsynaptic PKJs got a CF spike (error signal)
        posts_cf = cf_mask[post_idx]
        
        # 1) postsyn got CF (error signal), AND
        # 2) PF fired recently (within LTD window), AND  
        # 3) PF fired before CF (positive dt_pf means PF came first)
        ltd_mask = posts_cf & (dt_pf > 0) & (dt_pf <= ltd_win)
        if ltd_mask.any():
            w[ltd_mask] -= ltd_scale   # depression (decrease weight - learn to ignore this input)

    # --- LTP rule: PF must precede PKJ spike, *but only if no CF present* ---
    # This strengthens connections when PF input successfully drives PKJ without error signals
    if pkj_spikes.any():
        # only consider PKJs that spiked but did not get a CF (no error signal)
        posts_spike_no_cf = pkj_spikes[post_idx] & ~cf_mask[post_idx]
        
        if posts_spike_no_cf.any():
            # postsyn PKJs spiking without CF (successful firing)
            # 1) postsyn spiked without CF, AND
            # 2) within LTP window (timing matters!)
            ltp_mask = posts_spike_no_cf & (dt_pf > 0) & (dt_pf <= ltp_win)
            if ltp_mask.any():
                w[ltp_mask] += ltp_scale   # potentiation (increase weight - learn this input is useful)

    # --- No weight decay - CbmSim uses LTP/LTD equilibrium for forgetting ---
    # CbmSim doesn't use weight leak. Instead, the random walk forgetting happens because
    # PKJ and PF synapses happening in the LTP window work against the synapses happening
    # in the LTD window, creating an equilibrium that prevents weight drift.
    # decay_scale = cfg["plasticity_every_steps"] * cfg["dt"]  # time between plasticity updates
    # w = w * (1.0 - w_leak * decay_scale)  # REMOVED - CbmSim doesn't use this
    
    # --- Clip weights to min/max allowed values ---
    # Keep weights within realistic bounds
    w = cp.clip(w, w_min, w_max)

    return w, last_pf_spike, last_pkj_spike, last_cf_spike