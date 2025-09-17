import cupy as cp


def init_pf_state(N_pool, refrac_steps):
    """
    Initialize parallel fiber (PF) pool state.
    """
    return {
        "refrac": cp.zeros(N_pool, dtype=cp.int32),
        "refrac_steps": refrac_steps,
    }


def step_pf_coinflip(pf_state, rate_hz, dt):
    """
    Coin-flip firing for PF pool.
    """
    N = pf_state["refrac"].size
    pf_state["refrac"] = cp.maximum(pf_state["refrac"] - 1, 0)

    p_fire = rate_hz * dt
    rand = cp.random.random(N)
    pf_spikes = (rand < p_fire) & (pf_state["refrac"] == 0)

    pf_state["refrac"][pf_spikes] = pf_state["refrac_steps"]
    return pf_spikes


def draw_pf_conductances(N_conn, g_mean, g_std):
    """
    Draw random PF conductances.
    """
    return cp.random.normal(loc=g_mean, scale=g_std, size=N_conn).astype(cp.float32)


def step_mf_poisson(N_dcn, rate_hz, dt):
    """
    Generate mossy fiber (MF) spikes as Poisson process.
    """
    p_fire = rate_hz * dt
    rand = cp.random.random(N_dcn)
    return rand < p_fire