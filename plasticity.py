import cupy as cp

def update_pfpkj_plasticity(
    w, pre_idx, post_idx,
    pf_spikes, pkj_spikes, cf_mask,
    t, cfg,
    last_pf_spike, last_pkj_spike, last_cf_spike
):
    """
    Update synaptic weights for parallel fiber (PF) → Purkinje cell (PKJ) connections.
    Implements both LTP (long-term potentiation) and LTD (long-term depression).

    Inputs
    ------
    w : array
        Synaptic weights for PF→PKJ connections.
    pre_idx, post_idx : arrays
        Indices mapping each synapse to its presynaptic PF and postsynaptic PKJ.
    pf_spikes : bool array
        Which PFs spiked at this time step.
    pkj_spikes : bool array
        Which PKJs spiked at this time step.
    cf_mask : bool array
        Which PKJs received a climbing fiber (CF) spike this step (special LTD trigger).
    t : float
        Current simulation time (in steps).
    cfg : dict
        Simulation parameters (plasticity windows, scales, weight limits).
    last_pf_spike, last_pkj_spike, last_cf_spike : arrays
        Record of most recent spike times for each neuron type.
    """

    # --- Plasticity configuration parameters ---
    ltd_win   = cp.float32(cfg["ltd_window"]["t_pre_cf"])   # LTD window size
    ltp_win   = cp.float32(cfg["ltp_window"]["t_pre_cf"])   # LTP window size
    ltd_scale = cp.float32(cfg["ltd_scale"])                # How much weight decreases for LTD
    ltp_scale = cp.float32(cfg["ltp_scale"])                # How much weight increases for LTP
    w_min     = cp.float32(cfg["w_min"])                    # Lower bound on synaptic weight
    w_max     = cp.float32(cfg["w_max"])                    # Upper bound on synaptic weight

    # --- Update spike history for this step ---
    if pf_spikes.any():
        last_pf_spike = last_pf_spike.copy()      # copy to avoid in-place issues
        last_pf_spike[pf_spikes] = t              # record time for all spiking PFs
    if pkj_spikes.any():
        last_pkj_spike = last_pkj_spike.copy()
        last_pkj_spike[pkj_spikes] = t
    if cf_mask.any():
        cf_idx = cp.where(cf_mask)[0]             # which PKJs got CF this step
        last_cf_spike[cf_idx] = t

    # --- Compute how long ago each PF fired (per synapse) ---
    dt_pf = t - last_pf_spike[pre_idx]            # time since presyn PF spiked

    # --- LTD rule: PF must precede CF within window ---
    if cf_mask.any():
        # posts_cf marks synapses whose postsynaptic PKJs got a CF spike
        posts_cf = cf_mask[post_idx]
        # valid LTD when:
        # 1) postsyn got CF, AND
        # 2) PF spiked before current time, AND
        # 3) PF spike was within the LTD time window
        ltd_mask = posts_cf & (dt_pf > 0) & (dt_pf <= ltd_win)
        if ltd_mask.any():
            w = w.copy()
            w[ltd_mask] -= ltd_scale   # depression (reduce weight)

    # --- LTP rule: PF must precede PKJ spike, *but only if no CF present* ---
    if pkj_spikes.any():
        # only consider PKJs that spiked but did not get a CF
        pkj_valid = pkj_spikes & (~cf_mask)
        if pkj_valid.any():
            # postsyn PKJs spiking without CF
            posts_spike_no_cf = pkj_valid[post_idx]
            # valid LTP when:
            # 1) PF spike before PKJ spike, AND
            # 2) within LTP window
            ltp_mask = posts_spike_no_cf & (dt_pf > 0) & (dt_pf <= ltp_win)
            if ltp_mask.any():
                w = w.copy()
                w[ltp_mask] += ltp_scale   # potentiation (increase weight)

    # --- Clip weights to min/max allowed values ---
    w = cp.clip(w, w_min, w_max)

    return w, last_pf_spike, last_pkj_spike, last_cf_spike