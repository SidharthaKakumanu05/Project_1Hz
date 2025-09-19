import cupy as cp

def update_pfpkj_plasticity(
    w, pre_idx, post_idx,
    pf_spikes, pkj_spikes, cf_mask,
    t, cfg,
    last_pf_spike, last_pkj_spike, last_cf_spike
):
    """
    Vectorized PF->PKJ plasticity (LTP/LTD).
    """

    # --- Config params (match your config.py layout) ---
    ltd_win   = cp.float32(cfg["ltd_window"]["t_pre_cf"])
    ltp_win   = cp.float32(cfg["ltp_window"]["t_pre_cf"])
    ltd_scale = cp.float32(cfg["ltd_scale"])
    ltp_scale = cp.float32(cfg["ltp_scale"])
    w_min     = cp.float32(cfg["w_min"])
    w_max     = cp.float32(cfg["w_max"])

    # --- Update last spike times ---
    if pf_spikes.any():
        last_pf_spike = last_pf_spike.copy()
        last_pf_spike[pf_spikes] = t
    if pkj_spikes.any():
        last_pkj_spike = last_pkj_spike.copy()
        last_pkj_spike[pkj_spikes] = t
    if cf_mask.any():
        cf_idx = cp.where(cf_mask)[0]
        last_cf_spike[cf_idx] = t

    # --- Precompute PF recency per synapse ---
    dt_pf = t - last_pf_spike[pre_idx]

    # --- LTD: PF before CF ---
    if cf_mask.any():
        posts_cf = cf_mask[post_idx]                # synapses targeting PKJs that got CF
        ltd_mask = posts_cf & (dt_pf > 0) & (dt_pf <= ltd_win)
        if ltd_mask.any():
            w = w.copy()
            w[ltd_mask] -= ltd_scale

    # --- LTP: PF before PKJ (only if no CF) ---
    if pkj_spikes.any():
        pkj_valid = pkj_spikes & (~cf_mask)
        if pkj_valid.any():
            posts_spike_no_cf = pkj_valid[post_idx]
            ltp_mask = posts_spike_no_cf & (dt_pf > 0) & (dt_pf <= ltp_win)
            if ltp_mask.any():
                w = w.copy()
                w[ltp_mask] += ltp_scale

    # --- Clip weights ---
    w = cp.clip(w, w_min, w_max)

    return w, last_pf_spike, last_pkj_spike, last_cf_spike