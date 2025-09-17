import cupy as cp


class NeuronState:
    def __init__(self, N, lif_cfg):
        self.N = N
        self.V = cp.full(N, lif_cfg["EL"], dtype=cp.float32)  # membrane potentials
        self.refrac_timer = cp.zeros(N, dtype=cp.int32)
        self.spike = cp.zeros(N, dtype=bool)

    def reset(self, lif_cfg):
        self.V.fill(lif_cfg["EL"])
        self.refrac_timer.fill(0)
        self.spike.fill(False)


def lif_step(state, I_syn, I_ext, dt, lif_cfg):
    """
    One step of leaky integrate-and-fire dynamics.
    """
    V = state.V
    C = lif_cfg["C"]
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]

    # update refractory timers
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)

    # only update neurons not in refractory
    active = state.refrac_timer == 0
    dV = (-(V - EL) * gL + I_syn + I_ext) / C * dt
    V = V + dV * active

    # detect spikes
    spike = V >= Vth
    V = cp.where(spike, Vreset, V)

    # update state
    state.V = V
    state.spike = spike
    state.refrac_timer = cp.where(spike, lif_cfg["refrac_steps"], state.refrac_timer)

    return state