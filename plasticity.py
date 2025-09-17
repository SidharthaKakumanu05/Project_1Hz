import cupy as cp


def update_pfpkj_plasticity(
    w,                # cp.ndarray, shape [M] (one per PF→PKJ synapse)
    pre_idx,          # cp.ndarray, shape [M], PF indices for each synapse
    post_idx,         # cp.ndarray, shape [M], PKJ indices for each synapse
    pf_spikes,        # cp.ndarray(bool), shape [N_PF], PF spikes this step
    pkj_spikes,       # cp.ndarray(bool), shape [N_PKJ], (unused here but kept for API)
    cf_mask,          # cp.ndarray(bool), shape [N_PKJ], PKJs that got CF this step
    dt,               # float, timestep (s)
    cfg,              # dict, must include ltd_window/ltp_window + scales + w_min/w_max
    traces=None       # dict or None; keeps last spike times on GPU
):
    """
    PF→PKJ plasticity (CuPy, vectorized) with last-spike timing:

    - LTD (strong): PF spike within |Δt| ≤ LTD window (e.g., 0.1 s) of a CF on the same PKJ.
    - LTP (weak):   PF spike within LTP window (e.g., 0.9 s) but NOT in LTD window.

    Windows and scales from cfg:
      cfg["ltd_window"]["t_pre_cf"]  (e.g., 0.1)
      cfg["ltp_window"]["t_pre_cf"]  (e.g., 0.9)
      cfg["ltd_scale"]               (e.g., 9e-4)
      cfg["ltp_scale"]               (e.g., 1e-4)
      cfg["w_min"], cfg["w_max"]
    """

    # convert windows to step units
    ltd_win = int(cfg["ltd_window"]["t_pre_cf"] / dt)   # e.g., 0.1 s → steps
    ltp_win = int(cfg["ltp_window"]["t_pre_cf"] / dt)   # e.g., 0.9 s → steps
    ltd_scale = cp.float32(cfg["ltd_scale"])
    ltp_scale = cp.float32(cfg["ltp_scale"])

    # --- initialize traces if first call ---
    # We store LAST spike time (in integer steps) for each PF and PKJ (CF targets).
    # Sentinel = -1 (means "never").
    if traces is None:
        n_pf = int(cp.max(pre_idx).item()) + 1
        n_pkj = int(cp.max(post_idx).item()) + 1
        traces = {
            "pf_last": cp.full(n_pf, -1, dtype=cp.int64),
            "cf_last": cp.full(n_pkj, -1, dtype=cp.int64),
        }
        update_pfpkj_plasticity._t = 0  # simulation step counter

    pf_last = traces["pf_last"]
    cf_last = traces["cf_last"]

    # current simulation time in integer steps
    t = int(getattr(update_pfpkj_plasticity, "_t", 0))

    # --- update last-spike times from this step ---
    # PF that spiked now
    if pf_spikes.any():
        sp_pf_ids = cp.where(pf_spikes)[0]
        pf_last[sp_pf_ids] = t

    # PKJs that received a CF now (1:1 IO→CF→PKJ mapping already applied upstream)
    if cf_mask.any():
        sp_cf_ids = cp.where(cf_mask)[0]
        cf_last[sp_cf_ids] = t

    # --- vectorized timing lookups per synapse ---
    # For each synapse, look up the last PF time of its pre and last CF time of its post.
    pf_last_syn = pf_last[pre_idx]      # shape [M]
    cf_last_syn = cf_last[post_idx]     # shape [M]

    # Validity masks (spike has occurred at least once)
    pf_valid = pf_last_syn >= 0
    cf_valid = cf_last_syn >= 0

    # Δt in steps between the last PF and CF for that synapse's endpoints
    # (absolute for LTD window check)
    delta_pf_cf = cp.abs(pf_last_syn - cf_last_syn)

    # time since last PF for LTP window check
    dt_since_pf = t - pf_last_syn

    # time since last CF for excluding LTD-window cases in LTP
    dt_since_cf = t - cf_last_syn

    # --- LTD mask: PF & CF both valid and |Δt| ≤ ltd_win ---
    ltd_mask = pf_valid & cf_valid & (delta_pf_cf <= ltd_win)

    # --- LTP mask: PF valid & within ltp_win, but NOT in LTD window (and no CF within ltd_win) ---
    ltp_mask = (
        pf_valid
        & (dt_since_pf >= 0)
        & (dt_since_pf <= ltp_win)
        & (~ltd_mask)
        & ( (~cf_valid) | (dt_since_cf > ltd_win) )
    )

    # --- apply updates ---
    # Note: use in-place ops on GPU arrays
    w[ltd_mask] -= ltd_scale
    w[ltp_mask] += ltp_scale

    # clamp weights
    cp.clip(w, cfg["w_min"], cfg["w_max"], out=w)

    # advance global plasticity time
    update_pfpkj_plasticity._t = t + 1

    return w, traces