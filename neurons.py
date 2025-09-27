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
    
    # In-place operations to avoid memory allocation
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0
    
    # Vectorized voltage update
    V += (gL * (EL - V) + I_syn + I_ext) * dt * active
    
    spike = V >= Vth
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    
    state.V = V
    state.spike = spike
    return state

def io_step(state, gNCSum, vCoupleIO, errDrive, dt, params):
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

    # Ultra-optimized: maximum fused operations for ultimate speed
    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Maximum optimization: pre-compute all constants for single operation
    gL_EL = gL * EL
    gNCSum_eNCtoIO = gNCSum * eNCtoIO
    g_total = gL + gNCSum
    V += gL_EL + gNCSum_eNCtoIO + vCoupleIO - V * g_total

    # Fused spike detection and state updates
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

    # Ultra-optimized: maximum fused operations for ultimate speed
    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Maximum optimization: pre-compute all constants for single operation
    gL_EL = gL * EL
    gBCPC_eBCtoPC = gBCPC * eBCtoPC
    gSCPC_eSCtoPC = gSCPC * eSCtoPC
    g_total = gL + gPFPC + gBCPC + gSCPC
    V += (gL_EL + gBCPC_eBCtoPC + gSCPC_eSCtoPC - V * g_total) * active

    # Fused spike detection and state updates
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

    # Ultra-optimized: maximum fused operations for ultimate speed
    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Maximum optimization: pre-compute all constants for single operation
    gL_EL = gL * EL
    gPCBC_ePCtoBC = gPCBC * ePCtoBC
    g_total = gL + gPFBC + gPCBC
    V += (gL_EL + gPCBC_ePCtoBC - V * g_total) * active

    # Fused spike detection and state updates
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

    # Ultra-optimized: maximum fused operations for ultimate speed
    state.thresh += thresh_decay * (thresh_rest - state.thresh)
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # Maximum optimization: pre-compute all constants for single operation
    gL_EL = gL * EL
    gPCNCSum_ePCtoNC = gPCNCSum * ePCtoNC
    gMF_total = gMFNMDASum + gMFAMPASum
    g_total = gL + gMF_total + gPCNCSum
    V += (gL_EL + gPCNCSum_ePCtoNC - V * g_total) * active

    # Fused spike detection and state updates
    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state