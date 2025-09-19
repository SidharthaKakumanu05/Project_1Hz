import cupy as cp


class NeuronState:
    def __init__(self, N, lif_cfg):
        """
        Container for the state of a population of LIF neurons.

        Parameters
        ----------
        N : int
            Number of neurons in this population.
        lif_cfg : dict
            Parameters of the LIF model (from config).
        """
        self.N = N
        # Membrane voltages, initialized to leak reversal potential (EL)
        self.V = cp.full(N, lif_cfg["EL"], dtype=cp.float32)
        # Refractory countdown timer (0 = can fire, >0 = still waiting)
        self.refrac_timer = cp.zeros(N, dtype=cp.int32)
        # Boolean array: did each neuron spike this step?
        self.spike = cp.zeros(N, dtype=bool)

    def reset(self, lif_cfg):
        """Reset all neurons to rest state."""
        self.V.fill(lif_cfg["EL"])
        self.refrac_timer.fill(0)
        self.spike.fill(False)


def lif_step(state, I_syn, I_ext, dt, lif_cfg):
    """
    Advance the LIF model one timestep.

    Parameters
    ----------
    state : NeuronState
        Holds V, spike flags, refractory timers.
    I_syn : array
        Synaptic input current (from other neurons).
    I_ext : array
        External current (bias or injected).
    dt : float
        Timestep length (s).
    lif_cfg : dict
        Neuron parameters (C, gL, EL, Vth, Vreset, refrac_steps).

    Returns
    -------
    state : NeuronState
        Updated state after one step.
    """
    V = state.V
    C = lif_cfg["C"]
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]

    # --- Refractory update ---
    # Count down for neurons currently in refractory
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)

    # --- Voltage update for active neurons only ---
    active = state.refrac_timer == 0
    # dV = (leak + inputs) / capacitance * dt
    dV = (-(V - EL) * gL + I_syn + I_ext) / C * dt
    V = V + dV * active   # inactive (refractory) neurons get no update

    # --- Spike detection ---
    spike = V >= Vth              # any neuron above threshold spikes
    V = cp.where(spike, Vreset, V)  # reset spiking neurons to Vreset

    # --- Update state ---
    state.V = V
    state.spike = spike
    # neurons that spiked get a fresh refractory period
    state.refrac_timer = cp.where(spike, lif_cfg["refrac_steps"], state.refrac_timer)

    return state