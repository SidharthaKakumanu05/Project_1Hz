import cupy as cp


class SynapseProj:
    """
    Synapse projection class.

    Each object represents a connection type (e.g. PF→PKJ).
    It is responsible for:
    - keeping track of which presyn neurons connect to which postsyn neurons
    - applying weights to presynaptic spikes
    - implementing transmission delays
    - handling exponential decay of conductances
    """

    def __init__(self, pre_idx, post_idx, w_init, E_rev, tau, delay_steps):
        """
        Parameters
        ----------
        pre_idx : array
            Indices of presynaptic neurons (who sends signals).
        post_idx : array
            Indices of postsynaptic targets (who receives signals).
        w_init : array or scalar
            Initial synaptic weights/conductances.
        E_rev : float
            Reversal potential of the synapse (determines excitatory vs inhibitory).
        tau : float
            Time constant of exponential decay (seconds).
        delay_steps : int
            Transmission delay in number of timesteps.
        """
        # --- Connection wiring ---
        self.pre_idx = cp.asarray(pre_idx, dtype=cp.int32)
        self.post_idx = cp.asarray(post_idx, dtype=cp.int32)

        self.M = self.pre_idx.size                 # number of total synapses
        self.w = cp.asarray(w_init, dtype=cp.float32)  # synaptic weights

        # --- Synapse properties ---
        self.E_rev = cp.float32(E_rev)             # reversal potential
        self.tau = tau                             # decay time constant
        self.delay_steps = delay_steps             # how many steps to wait before effect

        # --- Delay buffer ---
        # Stores contributions that will be delivered in future steps.
        # Shape = (delay_steps+1, M), ring-buffer style.
        self.delay_buf = cp.zeros((delay_steps + 1, self.M), dtype=cp.float32)
        self.buf_ptr = 0   # pointer to "current slot"

        # --- Decay factor (set externally later) ---
        self.alpha = None

    def set_alpha(self, alpha):
        """
        Store exponential decay factor (alpha = exp(-dt/tau)).
        Typically set from simulate.py after dt is known.
        """
        self.alpha = alpha

    def enqueue_from_pre_spikes(self, pre_spikes, scale=None):
        """
        Add presynaptic spike contributions into the delay buffer.

        Parameters
        ----------
        pre_spikes : bool array
            Which presynaptic neurons spiked this step.
        scale : array or None
            Optional scaling factors (e.g. PF conductances).
        """
        # Convert presyn spikes into a float mask for all synapses
        active = pre_spikes[self.pre_idx].astype(cp.float32)

        # Apply optional scaling factors (e.g., sampled PF conductance per synapse)
        if scale is not None:
            active = active * scale

        # Multiply by synaptic weights to get final contribution
        contrib = active * self.w

        # Add contributions into the current buffer slot
        self.delay_buf[self.buf_ptr, :] += contrib

    def step_decay(self):
        """
        Advance the delay buffer pointer and reset the slot we just moved into.
        This acts like a circular buffer that rotates forward each timestep.
        """
        self.buf_ptr = (self.buf_ptr + 1) % self.delay_buf.shape[0]
        self.delay_buf[self.buf_ptr, :] *= 0.0  # clear this slot for reuse

    def currents_to_post(self):
        """
        Read out the effective conductance for each synapse at this step.

        Returns
        -------
        g : array
            Conductance values for all active synapses.
        post_idx : array
            Postsynaptic neuron index for each synapse (aligns with g).
        """
        # Sum contributions across delay buffer slots
        g = cp.sum(self.delay_buf, axis=0)
        return g, self.post_idx