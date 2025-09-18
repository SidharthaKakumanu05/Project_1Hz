import cupy as cp

def update_pfpkj_plasticity(
    w, pre_idx, post_idx,
    pf_spikes, pkj_spikes, cf_mask,
    t, cfg,
    last_pf_spike, last_pkj_spike, last_cf_spike
):
    """
    PF->PKJ plasticity using last-spike STDP rule.

    Parameters
    ----------
    w : cp.ndarray
        Synaptic weights for PF->PKJ.
    pre_idx : cp.ndarray
        Indices of presynaptic PFs.
    post_idx : cp.ndarray
        Indices of postsynaptic PKJs.
    pf_spikes : cp.ndarray (bool)
        Spikes from PF pool this step.
    pkj_spikes : cp.ndarray (bool)
        Spikes from PKJ cells this step.
    cf_mask : cp.ndarray (bool)
        Mask of PKJs that received CF input this step.
    t : float
        Current simulation time (s).
    cfg : dict
        Simulation config dictionary.
    last_pf_spike, last_pkj_spike, last_cf_spike : cp.ndarray
        Last spike times for PF, PKJ, and CF neurons.
    """

    ltd_win = cfg["ltd_window"]["t_pre_cf"]   # e.g. 0.05 (50 ms)
    ltp_win = cfg["ltp_window"]["t_pre_cf"]   # e.g. 0.45 (450 ms)

    ltd_scale = cfg["ltd_scale"]
    ltp_scale = cfg["ltp_scale"]

    # --- Update last spike times ---
    pf_spike_idx = cp.where(pf_spikes)[0]
    if pf_spike_idx.size > 0:
        last_pf_spike[pf_spike_idx] = t

    pkj_spike_idx = cp.where(pkj_spikes)[0]
    if pkj_spike_idx.size > 0:
        last_pkj_spike[pkj_spike_idx] = t

    cf_idx = cp.where(cf_mask)[0]
    if cf_idx.size > 0:
        last_cf_spike[cf_idx] = t

    # --- LTD: PF before CF ---
    for cf_post in cf_idx.tolist():
        pf_candidates = pre_idx[post_idx == cf_post]
        for pf in pf_candidates.tolist():
            dt = t - last_pf_spike[pf]
            if 0 < dt <= ltd_win:
                mask = (pre_idx == pf) & (post_idx == cf_post)
                w[mask] -= ltd_scale

    # --- LTP: PF before PKJ (if no CF this step) ---
    for pkj in pkj_spike_idx.tolist():
        if not cf_mask[pkj]:  # only if CF didn’t also arrive
            pf_candidates = pre_idx[post_idx == pkj]
            for pf in pf_candidates.tolist():
                dt = t - last_pf_spike[pf]
                if 0 < dt <= ltp_win:
                    mask = (pre_idx == pf) & (post_idx == pkj)
                    w[mask] += ltp_scale

    # --- Clip weights to valid range ---
    w = cp.clip(w, cfg["w_min"], cfg["w_max"])

    return w, last_pf_spike, last_pkj_spike, last_cf_spike