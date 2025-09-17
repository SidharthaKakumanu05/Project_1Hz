import cupy as cp


def init_pf_state(N_pool, refrac_steps, rng):
    """
    Initialize parallel fiber (PF) pool state.

    Parameters
    ----------
    N_pool : int
        Number of PFs in the pool.
    refrac_steps : int
        Refractory period (steps).
    rng : cp.random.Generator
        CuPy random generator.

    Returns
    -------
    pf_state : dict
        Tracks refractory timers.
    """
    return {
        "refrac": cp.zeros(N_pool, dtype=cp.int32),
        "refrac_steps": refrac_steps,
    }


def step_pf_coinflip(pf_state, rate_hz, dt, rng):
    """
    Coin-flip firing for PF pool.

    Each PF fires with probability = rate_hz * dt, provided it is not in refractory.

    Parameters
    ----------
    pf_state : dict
        State of PF pool (contains refractory counters).
    rate_hz : float
        Mean firing rate of each PF.
    dt : float
        Timestep (s).
    rng : cp.random.Generator

    Returns
    -------
    pf_spikes : cp.ndarray(bool)
        Spike array for PF pool (length = N_PF).
    """
    N = pf_state["refrac"].size

    # decrement refractory counters
    pf_state["refrac"] = cp.maximum(pf_state["refrac"] - 1, 0)

    # fire with probability rate*dt if not refractory
    p_fire = rate_hz * dt
    rand = rng.random(N)
    pf_spikes = (rand < p_fire) & (pf_state["refrac"] == 0)

    # reset refractory timers
    pf_state["refrac"][pf_spikes] = pf_state["refrac_steps"]

    return pf_spikes


def draw_pf_conductances(N_conn, g_mean, g_std, rng):
    """
    Draw random PF conductances.

    Parameters
    ----------
    N_conn : int
        Number of PF connections.
    g_mean : float
        Mean conductance.
    g_std : float
        Standard deviation of conductance.
    rng : cp.random.Generator

    Returns
    -------
    g : cp.ndarray
        Conductance array (length = N_conn).
    """
    return rng.normal(loc=g_mean, scale=g_std, size=N_conn).astype(cp.float32)


def step_mf_poisson(N_dcn, rate_hz, dt, rng):
    """
    Generate mossy fiber (MF) spikes as Poisson process.

    Parameters
    ----------
    N_dcn : int
        Number of DCN neurons (one MF per DCN here).
    rate_hz : float
        Mean MF firing rate.
    dt : float
        Timestep (s).
    rng : cp.random.Generator

    Returns
    -------
    mf_spikes : cp.ndarray(bool)
        MF spikes (length = N_dcn).
    """
    p_fire = rate_hz * dt
    rand = rng.random(N_dcn)
    return rand < p_fire