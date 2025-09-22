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

    def reset(self, lif_cfg):
        """
        Reset all neurons to their rest state.
        
        This is useful if you want to start over with a fresh population.
        """
        self.V.fill(lif_cfg["EL"])          # Reset voltage to resting potential
        self.refrac_timer.fill(0)           # Clear all refractory periods
        self.spike.fill(False)              # Clear all spike flags


def lif_step(state, I_syn, I_ext, dt, lif_cfg):
    """
    Advance the LIF model one timestep.
    
    This is the core function that simulates how neurons behave over time.
    It implements the Leaky Integrate-and-Fire model, which is based on:
    1. A neuron's membrane voltage changes based on incoming currents
    2. When voltage reaches threshold, the neuron "fires" (spikes)
    3. After firing, the neuron can't fire again for a short time (refractory period)
    
    The math: dV/dt = (leak current + synaptic currents + external current) / capacitance
    
    Parameters
    ----------
    state : NeuronState
        The current state of all neurons in this population (voltages, refractory timers, etc.)
    I_syn : array
        Synaptic input current from other neurons (positive = excitatory, negative = inhibitory)
    I_ext : array
        External current (bias current, injected current, etc.)
    dt : float
        Timestep length in seconds (how much time this step represents)
    lif_cfg : dict
        Neuron parameters from config.py (capacitance, conductance, thresholds, etc.)

    Returns
    -------
    state : NeuronState
        Updated state after one timestep - voltages, spike flags, refractory timers all updated
    """
    # Extract parameters for easier reading
    V = state.V                                    # Current membrane voltages
    C = lif_cfg["C"]                              # Membrane capacitance (how much charge it can hold)
    gL = lif_cfg["gL"]                            # Leak conductance (how fast voltage leaks away)
    EL = lif_cfg["EL"]                            # Leak reversal potential (resting voltage)
    Vth = lif_cfg["Vth"]                          # Spike threshold voltage
    Vreset = lif_cfg["Vreset"]                    # Voltage after a spike
    refrac_steps = lif_cfg["refrac_steps"]        # How long the refractory period lasts

    # --- Refractory update ---
    # Count down for neurons currently in refractory period
    # If a neuron just spiked, it can't spike again for a few time steps
    state.refrac_timer = cp.maximum(state.refrac_timer - 1, 0)

    # --- Voltage update for active neurons only ---
    # Only neurons that are NOT in refractory period can have their voltage change
    active = state.refrac_timer == 0
    
    # Calculate voltage change using the LIF equation:
    # dV = (leak current + synaptic inputs + external inputs) / capacitance * dt
    # Leak current pulls voltage back toward EL: -(V - EL) * gL
    dV = (-(V - EL) * gL + I_syn + I_ext) / C * dt
    V = V + dV * active   # Only active neurons get voltage updates

    # --- Spike detection ---
    # Any neuron whose voltage exceeds the threshold will spike
    spike = V >= Vth
    
    # Reset the voltage of spiking neurons back to Vreset
    V = cp.where(spike, Vreset, V)
    
    # Neurons that just spiked enter a refractory period
    state.refrac_timer = cp.where(spike, refrac_steps, state.refrac_timer)

    # --- Update state ---
    # Save the updated voltages and spike information back to the state object
    state.V = V
    state.spike = spike

    return state