
import numpy as np

class NeuronState:
    __slots__ = ("V","refrac","spike")
    def __init__(self, N, lif_cfg, rng):
        self.V = np.full(N, lif_cfg["Vreset"], dtype=np.float32)
        self.refrac = np.zeros(N, dtype=np.int32)
        self.spike = np.zeros(N, dtype=bool)

def lif_step(state, I_syn, I_ext, dt, lif_cfg):
    C = lif_cfg["C"]; gL = lif_cfg["gL"]; EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]; Vreset = lif_cfg["Vreset"]; refrac_steps = lif_cfg["refrac_steps"]
    V = state.V
    active = state.refrac <= 0
    V[~active] = Vreset
    state.refrac[~active] -= 1
    if np.any(active):
        idx = np.where(active)[0]
        dV = dt * ( -gL*(V[idx] - EL) + I_syn[idx] + I_ext[idx] ) / C
        V[idx] += dV
    spk = V >= Vth
    state.spike[:] = spk
    if np.any(spk):
        V[spk] = Vreset
        state.refrac[spk] = refrac_steps
    return state
