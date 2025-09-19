import cupy as cp

# -----------------------------
# PF state initialization
# -----------------------------
def init_pf_state(N_pool, refrac_steps):
    """
    Initialize the state dictionary for parallel fibers (PF).

    Each PF neuron has:
    - refrac: countdown timer (steps until it can fire again)
    - N: total number of PF neurons
    - refrac_steps: how many steps to stay silent after spiking
    """
    return {
        "refrac": cp.zeros(N_pool, dtype=cp.int32),  # all start ready-to-fire
        "N": N_pool,
        "refrac_steps": refrac_steps,
    }


# -----------------------------
# PF activity update (coinflip model)
# -----------------------------
def step_pf_coinflip(pf_state, rate_hz, dt):
    """
    Generate PF spikes with a coin-flip (Bernoulli) model.
    Each PF can fire with probability p = rate * dt,
    subject to refractory constraints.

    Returns a boolean array of spikes for this step.
    """
    p = rate_hz * dt                  # probability of firing in this timestep
    refrac = pf_state["refrac"]       # refractory countdown for each PF
    can_fire = refrac <= 0            # only neurons with 0 refractory can spike

    # Boolean array of spike events
    spikes = cp.zeros(pf_state["N"], dtype=bool)
    randu = cp.random.random(pf_state["N"])  # uniform random numbers
    spikes = cp.logical_and(can_fire, randu < p)

    # --- Update refractory state ---
    # neurons that spiked: set refractory timer
    refrac[spikes] = pf_state["refrac_steps"]
    # neurons that didn’t spike: decrement timer if > 0
    refrac[~spikes] = cp.maximum(0, refrac[~spikes] - 1)

    pf_state["refrac"] = refrac       # save back updated refractory
    return spikes


# -----------------------------
# Draw PF synaptic conductances
# -----------------------------
def draw_pf_conductances(N_conn, g_mean, g_std):
    """
    Sample conductance strengths for PF connections.
    Each connection gets a value drawn from a Gaussian distribution.

    Negative draws are clipped to 0 (no negative conductance).
    """
    g = cp.random.normal(loc=g_mean, scale=g_std, size=N_conn).astype(cp.float32)
    return cp.maximum(g, 0.0)


# -----------------------------
# Mossy fiber spikes (Poisson model)
# -----------------------------
def step_mf_poisson(N, rate_hz, dt):
    """
    Generate Poisson spikes for mossy fibers (MF).

    Each of the N mossy fibers has probability p = rate * dt
    of firing in this timestep.
    """
    p = rate_hz * dt
    return (cp.random.random(N) < p)