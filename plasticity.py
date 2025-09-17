import cupy as cp


def update_pfpkj_plasticity(weights, pre_idx, post_idx,
                            pf_spikes, pkj_spikes, cf_mask,
                            dt, cfg, traces=None):
    """
    STDP-like PF→PKJ plasticity.
    - LTD: narrow, strong (~100 ms window)
    - LTP: broad, weak (~900 ms window)
    cf_mask: bool mask of PKJs receiving CF spikes at this timestep
    """

    if traces is None:
        traces = {
            "last_pf_spike": cp.full(pre_idx.max() + 1, -cp.inf, dtype=cp.float32),
            "last_pkj_spike": cp.full(post_idx.max() + 1, -cp.inf, dtype=cp.float32),
        }

    t_now = cp.float32(traces.get("t", 0.0))

    # update last-spike times
    pf_idx_fired = cp.where(pf_spikes)[0]
    if pf_idx_fired.size > 0:
        traces["last_pf_spike"][pf_idx_fired] = t_now

    pkj_idx_fired = cp.where(pkj_spikes)[0]
    if pkj_idx_fired.size > 0:
        traces["last_pkj_spike"][pkj_idx_fired] = t_now

    # LTD: PF active recently + CF arrives
    ltd_window = cfg["ltd_window"]["t_pre_cf"]
    ltp_window = cfg["ltp_window"]["t_pre_cf"]
    ltd_scale  = cfg["ltd_scale"]
    ltp_scale  = cfg["ltp_scale"]

    if cf_mask.any():
        pkj_targets = cp.where(cf_mask)[0]
        for pkj in pkj_targets:
            # find PFs connected to this PKJ
            mask = (post_idx == pkj)
            pf_connected = pre_idx[mask]

            # LTD if PF fired within ltd_window
            last_pf = traces["last_pf_spike"][pf_connected]
            dt_pf = t_now - last_pf
            eligible = (dt_pf >= 0) & (dt_pf <= ltd_window)
            weights[mask][eligible] -= ltd_scale

    # LTP: PF fired, PKJ spiked later (broad window)
    if pkj_idx_fired.size > 0:
        for pkj in pkj_idx_fired:
            mask = (post_idx == pkj)
            pf_connected = pre_idx[mask]
            last_pf = traces["last_pf_spike"][pf_connected]
            dt_pf = t_now - last_pf
            eligible = (dt_pf >= 0) & (dt_pf <= ltp_window)
            weights[mask][eligible] += ltp_scale

    # clip weights
    weights = cp.clip(weights, cfg["w_min"], cfg["w_max"])

    traces["t"] = t_now + dt
    return weights, traces