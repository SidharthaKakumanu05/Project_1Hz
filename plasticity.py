import numpy as np

def update_pfpkj_plasticity(w, pre_idx, post_idx,
                            pf_spikes, pkj_spikes, cf_mask,
                            dt, cfg, traces=None):
    """
    PF→PKJ plasticity with last-spike tracking.

    - LTD: PF spike within 100 ms of CF spike (strong depression).
    - LTP: PF spike within 900 ms of CF spike but outside LTD window (weak potentiation).
    """

    # window sizes in steps
    ltd_win  = int(cfg["ltd_window"]["t_pre_cf"] / dt)   # 100 ms
    ltp_win  = int(cfg["ltp_window"]["t_pre_cf"] / dt)   # 900 ms
    ltd_scale = cfg["ltd_scale"]
    ltp_scale = cfg["ltp_scale"]

    # initialize traces if first call
    if traces is None:
        traces = {
            "pf_last": np.full(pre_idx.max() + 1, -np.inf),   # last PF spike time (steps)
            "cf_last": np.full(post_idx.max() + 1, -np.inf)   # last CF spike time (steps)
        }
        update_pfpkj_plasticity._t = 0  # simulation time in steps

    pf_last = traces["pf_last"]
    cf_last = traces["cf_last"]

    # current simulation time (in steps)
    t = getattr(update_pfpkj_plasticity, "_t", 0)

    # update traces
    pf_ids = np.where(pf_spikes)[0]
    if pf_ids.size > 0:
        pf_last[pf_ids] = t

    cf_ids = np.where(cf_mask)[0]
    if cf_ids.size > 0:
        cf_last[cf_ids] = t

    # loop over synapses
    for syn in range(w.size):
        pre = pre_idx[syn]
        post = post_idx[syn]

        # time since PF and CF
        dt_pf = t - pf_last[pre]
        dt_cf = t - cf_last[post]

        # LTD: PF within 100 ms of CF
        if dt_pf >= 0 and abs(pf_last[pre] - cf_last[post]) <= ltd_win:
            w[syn] -= ltd_scale

        # LTP: PF within 900 ms window, but not in LTD window
        elif dt_pf >= 0 and dt_pf <= ltp_win and (t - cf_last[post]) > ltd_win:
            w[syn] += ltp_scale

    # clamp weights
    np.clip(w, cfg["w_min"], cfg["w_max"], out=w)

    # advance time
    update_pfpkj_plasticity._t = t + 1

    return w, traces