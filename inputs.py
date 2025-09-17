
import numpy as np

def init_pf_state(N_pool, refrac_steps, rng):
    return {"refrac": np.zeros(N_pool, dtype=np.int32),
            "active": np.zeros(N_pool, dtype=bool),
            "N": N_pool,
            "refrac_steps": refrac_steps}

def step_pf_coinflip(state, rate_hz, dt, rng):
    N = state["N"]; p = rate_hz * dt
    can_fire = state["refrac"] <= 0
    fire = rng.random(N) < p
    spike = fire & can_fire
    state["refrac"][spike] = state["refrac_steps"]
    state["refrac"][~spike & ~can_fire] -= 1
    state["active"][:] = spike
    return spike

def draw_pf_conductances(M, mean, std, rng):
    g = rng.normal(loc=mean, scale=std, size=M).astype(np.float32)
    g[g<0] = mean*0.5
    return g

def step_mf_poisson(N_dcn, rate_hz, dt, rng):
    return rng.random(N_dcn) < (rate_hz * dt)
