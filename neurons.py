import cupy as cp

class NeuronState:
    def __init__(self, N, params):
        self.N = N
        self.V = cp.full(N, params["EL"], dtype=cp.float32)
        self.refrac_timer = cp.zeros(N, dtype=cp.int32)
        self.spike = cp.zeros(N, dtype=bool)
        self.thresh = cp.full(N, params["Vth"], dtype=cp.float32)

    def reset(self, params):
        self.V.fill(params["EL"])
        self.refrac_timer.fill(0)
        self.spike.fill(False)
        self.thresh.fill(params["Vth"])

def lif_step(state, I_syn, I_ext, dt, params):
    V = state.V
    gL = params["gL"]
    EL = params["EL"]
    Vth = params["Vth"]
    Vreset = params["Vreset"]
    refrac_steps = params["refrac_steps"]
    
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0
    
    V += (gL * (EL - V) + I_syn + I_ext) * dt * active
    
    spike = V >= Vth
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    
    state.V = V
    state.spike = spike
    return state

def io_step(state, gNCSum, vCoupleIO, gNoise, errDrive, dt, params):
    V = state.V
    gL = params["gL"]
    EL = params["EL"]
    Vth = params["Vth"]
    Vreset = params["Vreset"]
    refrac_steps = params["refrac_steps"]
    thresh_decay = params["thresh_decay"]
    thresh_rest = params["thresh_rest"]
    eNCtoIO = params["eNCtoIO"]
    thresh_max = params["thresh_max"]
    
    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Fixed voltage update to match CbmSim: gL*(EL-V) + gNCSum*(eNCtoIO-V) + vCoupleIO + gNoise (removed errDrive)
    V += gL*(EL-V) + gNCSum*(eNCtoIO-V) + vCoupleIO + gNoise

    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state

def pkj_step(state, gPFPC, gBCPC, gSCPC, dt, params):
    V = state.V
    gL = params["gL"]
    EL = params["EL"]
    Vth = params["Vth"]
    Vreset = params["Vreset"]
    refrac_steps = params["refrac_steps"]
    thresh_decay = params["thresh_decay"]
    thresh_rest = params["thresh_rest"]
    thresh_max = params["thresh_max"]
    eBCtoPC = params["eBCtoPC"]
    eSCtoPC = params["eSCtoPC"]

    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Fixed voltage update to match CbmSim: gL*(EL-V) - gPFPC*V + gBCPC*(eBCtoPC-V) + gSCPC*(eSCtoPC-V)
    V += (gL * (EL - V) - gPFPC * V + gBCPC * (eBCtoPC - V) + gSCPC * (eSCtoPC - V)) * active

    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state

def bc_step(state, gPFBC, gPCBC, dt, params):
    V = state.V
    gL = params["gL"]
    EL = params["EL"]
    Vth = params["Vth"]
    Vreset = params["Vreset"]
    refrac_steps = params["refrac_steps"]
    thresh_decay = params["thresh_decay"]
    thresh_rest = params["thresh_rest"]
    thresh_max = params["thresh_max"]
    ePCtoBC = params["ePCtoBC"]

    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Fixed voltage update to match CbmSim: gL*(EL-V) - gPFBC*V + gPCBC*(ePCtoBC-V)
    V += (gL * (EL - V) - gPFBC * V + gPCBC * (ePCtoBC - V)) * active

    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state

def dcn_step(state, gMFNMDASum, gMFAMPASum, gPCNCSum, dt, params):
    V = state.V
    gL = params["gL"]
    EL = params["EL"]
    Vth = params["Vth"]
    Vreset = params["Vreset"]
    refrac_steps = params["refrac_steps"]
    thresh_decay = params["thresh_decay"]
    thresh_rest = params["thresh_rest"]
    thresh_max = params["thresh_max"]
    ePCtoNC = params["ePCtoNC"]

    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Fixed voltage update to match CbmSim: gL*(EL-V) - (gMFNMDASum + gMFAMPASum)*V + gPCNCSum*(ePCtoNC-V)
    V += (gL * (EL - V) - (gMFNMDASum + gMFAMPASum) * V + gPCNCSum * (ePCtoNC - V)) * active

    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state