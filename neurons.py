#!/usr/bin/env python3
"""
Neuron models and state management for the cerebellar microcircuit simulation.

This file implements the Leaky Integrate-and-Fire (LIF) neuron model and manages
the state of neuron populations. The LIF model is a simplified but realistic
representation of how real neurons behave.

For a freshman undergrad: This is where we define what a "neuron" looks like
and how it behaves when it receives electrical signals!
"""

import cupy as cp


class NeuronState:
    """
    Container for the state of a population of LIF neurons.
    
    This class holds all the information we need to track about a group of neurons:
    - Their membrane voltages (like the electrical charge inside each neuron)
    - Whether they're in a refractory period (can't spike again immediately)
    - Whether they spiked in the current time step
    
    Think of this as a "data structure" that keeps track of all the neurons in one population.
    """
    
    def __init__(self, N, lif_cfg):
        """
        Initialize a population of N neurons with the given LIF parameters.

        Parameters
        ----------
        N : int
            Number of neurons in this population (e.g., 128 for Purkinje cells)
        lif_cfg : dict
            Parameters of the LIF model (from config.py) - things like capacitance,
            leak conductance, threshold voltage, etc.
        """
        self.N = N
        # Membrane voltages, initialized to leak reversal potential (EL)
        # This is like the "resting voltage" of each neuron - where they start
        self.V = cp.full(N, lif_cfg["EL"], dtype=cp.float32)
        
        # Refractory countdown timer (0 = can fire, >0 = still waiting)
        # After a neuron spikes, it can't spike again for a short time (refractory period)
        self.refrac_timer = cp.zeros(N, dtype=cp.int32)
        
        # Boolean array: did each neuron spike this step?
        # This tells us which neurons fired action potentials in the current time step
        self.spike = cp.zeros(N, dtype=bool)
        
        # Dynamic thresholds (CbmSim style) - initialized to static threshold
        self.thresh = cp.full(N, lif_cfg["Vth"], dtype=cp.float32)

    def reset(self, lif_cfg):
        """
        Reset all neurons to their rest state.
        
        This is useful if you want to start over with a fresh population.
        """
        self.V.fill(lif_cfg["EL"])          # Reset voltage to resting potential
        self.refrac_timer.fill(0)           # Clear all refractory periods
        self.spike.fill(False)              # Clear all spike flags
        self.thresh.fill(lif_cfg["Vth"])    # Reset thresholds to static value


def lif_step(state, I_syn, I_ext, dt, lif_cfg):
    """
    Generic LIF neuron step function for non-CbmSim neurons.
    
    This is a simplified LIF model for neurons that don't need CbmSim's exact dynamics.
    It implements the basic leaky integrate-and-fire equations.
    
    Parameters
    ----------
    state : NeuronState
        Current neuron state (V, refrac_timer, spike)
    I_syn : array
        Synaptic current input
    I_ext : array
        External current input (bias, etc.)
    dt : float
        Time step size
    lif_cfg : dict
        LIF neuron parameters (gL, EL, Vth, Vreset, refrac_steps)
    
    Returns
    -------
    state : NeuronState
        Updated neuron state
    """
    V = state.V
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]
    refrac_steps = lif_cfg["refrac_steps"]
    
    # Update refractory timer
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0
    
    # LIF equation: dV/dt = (gL * (EL - V) + I_syn + I_ext) / C
    # For simplicity, assume C = 1 (normalized)
    V += (gL * (EL - V) + I_syn + I_ext) * dt * active
    
    # Spike detection
    spike = V >= Vth
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    
    state.V = V
    state.spike = spike
    return state


def cbmsim_io_step(state, gNCSum, vCoupleIO, gNoise, dt, lif_cfg):
    """
    CbmSim exact IO neuron update: vIO += gLeakIO * (eLeakIO - vIO) + gNCSum * (eNCtoIO - vIO) + vCoupleIO + gNoise
    No error drive - removed for long-term stability testing
    """
    V = state.V
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]
    refrac_steps = lif_cfg["refrac_steps"]
    thresh_decay = lif_cfg["thresh_decay"]
    thresh_rest = lif_cfg["thresh_rest"]
    eNCtoIO = lif_cfg["eNCtoIO"]

    # Update dynamic threshold (CbmSim: threshIO[i] += threshDecIO * (threshRestIO - threshIO[i]))
    state.thresh += thresh_decay * (thresh_rest - state.thresh)

    # Refractory update
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # CbmSim exact IO equation - NO capacitance, NO dt multiplication!
    V += (gL * (EL - V) + gNCSum * (eNCtoIO - V) + vCoupleIO + gNoise[0]) * active

    # Spike detection
    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    # CbmSim threshold reset: set to max if we spiked, otherwise keep same
    thresh_max = lif_cfg["thresh_max"]
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state

def cbmsim_pc_step(state, gPFPC, gBCPC, gSCPC, dt, lif_cfg):
    """
    CbmSim exact PC neuron update: vPC += gLeakPC * (eLeakPC - vPC) - gPFPC * vPC + gBCPC * (eBCtoPC - vPC) + gSCPC * (eSCtoPC - vPC)
    """
    V = state.V
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]
    refrac_steps = lif_cfg["refrac_steps"]
    thresh_decay = lif_cfg["thresh_decay"]
    thresh_rest = lif_cfg["thresh_rest"]
    eBCtoPC = lif_cfg["eBCtoPC"]
    eSCtoPC = lif_cfg["eSCtoPC"]

    # Update dynamic threshold
    state.thresh += thresh_decay * (thresh_rest - state.thresh)

    # Refractory update
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # FIXED PC equation - use proper synaptic currents instead of conductance*voltage
    V += (gL * (EL - V) + gPFPC + gBCPC + gSCPC) * active

    # Spike detection
    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    # CbmSim threshold reset: set to max if we spiked, otherwise keep same
    thresh_max = lif_cfg["thresh_max"]
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state

def cbmsim_bc_step(state, gPFBC, gPCBC, dt, lif_cfg):
    """
    CbmSim exact BC neuron update: vBC += gLeakBC * (eLeakBC - vBC) - gPFBC * vBC + gPCBC * (ePCtoBC - vBC)
    """
    V = state.V
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]
    refrac_steps = lif_cfg["refrac_steps"]
    thresh_decay = lif_cfg["thresh_decay"]
    thresh_rest = lif_cfg["thresh_rest"]
    ePCtoBC = lif_cfg["ePCtoBC"]

    # Update dynamic threshold
    state.thresh += thresh_decay * (thresh_rest - state.thresh)

    # Refractory update
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # FIXED BC equation - use proper synaptic currents instead of conductance*voltage
    V += (gL * (EL - V) + gPFBC + gPCBC) * active

    # Spike detection
    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    # CbmSim threshold reset: set to max if we spiked, otherwise keep same
    thresh_max = lif_cfg["thresh_max"]
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state

def cbmsim_nc_step(state, gMFNMDASum, gMFAMPASum, gPCNCSum, dt, lif_cfg):
    """
    CbmSim exact NC neuron update: vNC += gLeakNC * (eLeakNC - vNC) - (gMFNMDASum + gMFAMPASum) * vNC + gPCNCSum * (ePCtoNC - vNC)
    """
    V = state.V
    gL = lif_cfg["gL"]
    EL = lif_cfg["EL"]
    Vth = lif_cfg["Vth"]
    Vreset = lif_cfg["Vreset"]
    refrac_steps = lif_cfg["refrac_steps"]
    thresh_decay = lif_cfg["thresh_decay"]
    thresh_rest = lif_cfg["thresh_rest"]
    ePCtoNC = lif_cfg["ePCtoNC"]

    # Update dynamic threshold
    state.thresh += thresh_decay * (thresh_rest - state.thresh)

    # Refractory update
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)
    active = state.refrac_timer == 0

    # FIXED NC equation - use proper synaptic currents instead of conductance*voltage
    V += (gL * (EL - V) + gMFNMDASum + gMFAMPASum + gPCNCSum) * active

    # Spike detection
    spike = V >= state.thresh
    V = cp.where(spike, Vreset, V)
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)
    # CbmSim threshold reset: set to max if we spiked, otherwise keep same
    thresh_max = lif_cfg["thresh_max"]
    state.thresh = cp.where(spike, thresh_max, state.thresh)

    state.V = V
    state.spike = spike
    return state