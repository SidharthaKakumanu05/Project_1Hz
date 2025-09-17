import cupy as cp


class NeuronState:
    """
    Holds the state of a population of leaky integrate-and-fire neurons.
    All arrays are allocated on GPU using CuPy.
    """

    def __init__(self, N, lif_cfg, rng):
        """
        Parameters
        ----------
        N : int
            Number of neurons in this population.
        lif_cfg : dict
            Dictionary of LIF parameters (C, gL, EL, Vth, Vreset, refrac_steps).
        rng : cupy.random.Generator
            Random generator for this population.
        """
        self.V = cp.full(N, lif_cfg["EL"], dtype=cp.float32)       # membrane potential
        self.refrac = cp.zeros(N, dtype=cp.int32)                  # refractory counter
        self.spike = cp.zeros(N, dtype=bool)                       # spike indicator
        self.rng = rng                                             # CuPy RNG


def lif_step(state, I_syn, I_ext, dt, lif_cfg):
    """
    Advance the LIF neuron state by one timestep on GPU.

    Parameters
    ----------
    state : NeuronState
        Current state of the neurons.
    I_syn : cp.ndarray
        Synaptic input current for each neuron.
    I_ext : cp.ndarray
        External bias/input current for each neuron.
    dt : float
        Simulation timestep.
    lif_cfg : dict
        LIF parameters.

    Returns
    -------
    NeuronState
        Updated neuron state.
    """
    V = state.V
    refrac = state.refrac

    # spikes if above threshold and not refractory
    spike = (V >= lif_cfg["Vth"]) & (refrac == 0)

    # integrate LIF membrane equation
    dV = (lif_cfg["EL"] - V) * lif_cfg["gL"] / lif_cfg["C"] + (I_syn + I_ext) / lif_cfg["C"]
    V = V + dt * dV

    # reset voltage after spike
    V[spike] = lif_cfg["Vreset"]

    # update refractory counters
    refrac[refrac > 0] -= 1
    refrac[spike] = lif_cfg["refrac_steps"]

    # save back
    state.V = V
    state.refrac = refrac
    state.spike = spike

    return state