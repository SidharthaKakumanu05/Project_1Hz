#!/usr/bin/env python3
"""
Synaptic connections and transmission for the cerebellar microcircuit simulation.

This file implements the SynapseProj class, which manages how signals are transmitted
from one group of neurons to another. It handles synaptic delays, weights, and
exponential decay of synaptic effects.

For a freshman undergrad: This is where we define how neurons "talk" to each other!
When one neuron spikes, it sends a signal to other neurons - this file manages that process.
"""

import cupy as cp


class SynapseProj:
    """
    Synapse projection class - manages connections between neuron populations.

    Each object represents one type of connection (e.g., PF→PKJ, BC→PKJ, etc.).
    It handles all the complex details of synaptic transmission:
    - Which neurons connect to which other neurons (the "wiring")
    - How strong each connection is (synaptic weights)
    - How long it takes for signals to travel (synaptic delays)
    - How long synaptic effects last (exponential decay)
    
    Think of this as the "postal service" between neurons - it makes sure signals
    get delivered to the right place, at the right time, with the right strength.
    """

    def __init__(self, pre_idx, post_idx, w_init, E_rev, tau, delay_steps):
        """
        Initialize a synaptic projection between two neuron populations.
        
        This sets up the "wiring" between neurons and defines how signals are transmitted.

        Parameters
        ----------
        pre_idx : array
            Indices of presynaptic neurons (who sends signals) - the "senders"
        post_idx : array
            Indices of postsynaptic targets (who receives signals) - the "receivers"
        w_init : array or scalar
            Initial synaptic weights/conductances - how strong each connection is
        E_rev : float
            Reversal potential of the synapse (determines excitatory vs inhibitory)
            E_rev = 0: excitatory, E_rev < 0: inhibitory
        tau : float
            Time constant of exponential decay (seconds) - how long effects last
        delay_steps : int
            Transmission delay in number of timesteps - how long signals take to travel
        """
        # --- Connection wiring ---
        # These arrays define which neurons connect to which other neurons
        # For example, if pre_idx[0] = 5 and post_idx[0] = 12, then neuron 5 connects to neuron 12
        self.pre_idx = cp.asarray(pre_idx, dtype=cp.int32)   # Sender neuron indices
        self.post_idx = cp.asarray(post_idx, dtype=cp.int32) # Receiver neuron indices

        self.M = self.pre_idx.size                 # Total number of synapses (connections)
        self.w = cp.asarray(w_init, dtype=cp.float32)  # Synaptic weights (connection strengths)

        # --- Synapse properties ---
        self.E_rev = cp.float32(E_rev)             # Reversal potential (excitatory vs inhibitory)
        self.tau = tau                             # Decay time constant (how long effects last)
        self.delay_steps = delay_steps             # Transmission delay (how long signals take to travel)

        # --- Delay buffer ---
        # This is a clever way to handle synaptic delays without storing lots of history
        # It's like a circular buffer that rotates each timestep
        # Shape = (delay_steps+1, M), where each row represents one time step of delay
        self.delay_buf = cp.zeros((delay_steps + 1, self.M), dtype=cp.float32)
        self.buf_ptr = 0   # Pointer to the "current slot" in the circular buffer

        # --- Decay factor (set externally later) ---
        # This will be calculated from tau and dt to make decay calculations faster
        self.alpha = None

    def set_alpha(self, alpha):
        """
        Store exponential decay factor (alpha = exp(-dt/tau)).
        
        This is a precomputed value that makes the decay calculations faster.
        It's typically set from simulate.py after the timestep (dt) is known.
        
        Parameters
        ----------
        alpha : float
            The exponential decay factor for this synapse type
        """
        self.alpha = alpha

    def enqueue_from_pre_spikes(self, pre_spikes, scale=None):
        """
        Add presynaptic spike contributions into the delay buffer.
        
        This is where the "magic" happens - when presynaptic neurons spike,
        this function calculates how much effect they will have on postsynaptic
        neurons (after the delay) and stores it in the delay buffer.

        Parameters
        ----------
        pre_spikes : bool array
            Which presynaptic neurons spiked this step (True = spiked, False = didn't spike)
        scale : array or None
            Optional scaling factors (e.g., PF conductances that vary per connection)
        """
        # Convert presyn spikes into a float mask for all synapses
        # This tells us which synapses are "active" because their presynaptic neuron spiked
        active = pre_spikes[self.pre_idx].astype(cp.float32)

        # Apply optional scaling factors (e.g., sampled PF conductance per synapse)
        # This allows each connection to have a different strength even if weights are the same
        if scale is not None:
            active = active * scale

        # Multiply by synaptic weights to get final contribution
        # This gives us the actual "strength" of the signal that will be delivered
        contrib = active * self.w

        # Add contributions into the current buffer slot
        # These contributions will be "delivered" after the synaptic delay
        self.delay_buf[self.buf_ptr, :] += contrib

    def step_decay(self):
        """
        Advance the delay buffer pointer and reset the slot we just moved into.
        
        This function is called every timestep to "rotate" the delay buffer.
        It's like moving the hands of a clock forward - signals that were scheduled
        to arrive later are now one step closer to being delivered.
        
        The delay buffer works like a circular queue:
        - Each row represents one time step of delay
        - Each timestep, we move to the next row
        - When we reach the end, we wrap around to the beginning
        """
        # Move the pointer to the next slot in the circular buffer
        self.buf_ptr = (self.buf_ptr + 1) % self.delay_buf.shape[0]
        # Clear the slot we just moved into (faster than multiplying by decay factor)
        self.delay_buf[self.buf_ptr, :] = 0.0

    def currents_to_post(self):
        """
        Read out the effective conductance for each synapse at this step.
        
        This function calculates how much current each postsynaptic neuron
        should receive from this synaptic projection. It sums up all the
        delayed signals that are ready to be delivered now.

        Returns
        -------
        g : array
            Conductance values for all active synapses (how strong each signal is)
        post_idx : array
            Postsynaptic neuron index for each synapse (which neuron receives each signal)
        """
        # Sum contributions across all delay buffer slots
        # This gives us the total effect of all delayed signals that are ready now
        g = cp.sum(self.delay_buf, axis=0)
        return g, self.post_idx