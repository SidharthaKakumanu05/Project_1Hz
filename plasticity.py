import cupy as cp

def update_pfpkj_plasticity(
    w, pre_idx, post_idx,
    pf_spikes, pkj_spikes, cf_mask,
    t, cfg,
    last_pf_spike, last_pkj_spike, last_cf_spike
):
    ltd_win = cp.float32(cfg["ltd_window"])
    ltp_win = cp.float32(cfg["ltp_window"])
    ltd_scale = cp.float32(cfg["ltd_scale"])
    ltp_scale = cp.float32(cfg["ltp_scale"])
    w_min = cp.float32(cfg["weight_min"])
    w_max = cp.float32(cfg["weight_max"])

    if pf_spikes.any():
        last_pf_spike[pf_spikes] = t
    if pkj_spikes.any():
        last_pkj_spike[pkj_spikes] = t
    if cf_mask.any():
        cf_idx = cp.where(cf_mask)[0]
        last_cf_spike[cf_idx] = t

    dt_pf = t - last_pf_spike[pre_idx]

    if cf_mask.any():
        posts_cf = cf_mask[post_idx]
        # LTD condition: PF spiked within ltd_win BEFORE CF spike
        # dt_pf > 0 means PF spiked before current time (which is when CF spikes)
        ltd_mask = posts_cf & (dt_pf > 0) & (dt_pf <= ltd_win)
        if ltd_mask.any():
            w[ltd_mask] += ltd_scale  # ltd_scale is already negative (-0.00275)

    if pkj_spikes.any():
        posts_spike_no_cf = pkj_spikes[post_idx] & ~cf_mask[post_idx]
        
        if posts_spike_no_cf.any():
            # LTP condition: PF spiked within ltp_win BEFORE PKJ spike (without CF)
            ltp_mask = posts_spike_no_cf & (dt_pf > 0) & (dt_pf <= ltp_win)
            if ltp_mask.any():
                w[ltp_mask] += ltp_scale

    w = cp.clip(w, w_min, w_max)

    return w, last_pf_spike, last_pkj_spike, last_cf_spike