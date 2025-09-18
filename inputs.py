import cupy as cp

def init_pf_state(N_pool, refrac_steps):
    return {
        "refrac": cp.zeros(N_pool, dtype=cp.int32),
        "N": N_pool,
        "refrac_steps": refrac_steps,
    }

def step_pf_coinflip(pf_state, rate_hz, dt):
    """Bernoulli coinflip PF spikes with refractory."""
    p = rate_hz * dt
    refrac = pf_state["refrac"]
    can_fire = refrac <= 0
    spikes = cp.zeros(pf_state["N"], dtype=bool)
    randu = cp.random.random(pf_state["N"])
    spikes = cp.logical_and(can_fire, randu < p)
    # update refractory: set after spike, decrement otherwise
    refrac[spikes] = pf_state["refrac_steps"]
    refrac[~spikes] = cp.maximum(0, refrac[~spikes] - 1)
    pf_state["refrac"] = refrac
    return spikes

def draw_pf_conductances(N_conn, g_mean, g_std):
    g = cp.random.normal(loc=g_mean, scale=g_std, size=N_conn).astype(cp.float32)
    return cp.maximum(g, 0.0)  # clip at 0

def step_mf_poisson(N, rate_hz, dt):
    """Poisson MF spikes of length N (must match MF_to_DCN pre pool)."""
    p = rate_hz * dt
    return (cp.random.random(N) < p)
