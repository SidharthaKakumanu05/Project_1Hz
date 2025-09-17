import cupy as cp


class SynapseProj:
    """
    Synapse projection class for handling connectivity, conductance,
    and current delivery between pre and post neuron populations.
    """

    def __init__(self, pre_idx, post_idx, w_init, E_rev, tau, delay_steps, rng):
        """
        Parameters
        ----------
        pre_idx : array-like
            Indices of presynaptic neurons.
        post_idx : array-like
            Indices of postsynaptic neurons (aligned with pre_idx).
        w_init : array-like
            Initial synaptic weights.
        E_rev : float
            Reversal potential of the synapse.
        tau : float
            Synaptic time constant.
        delay_steps : int
            Transmission delay in timesteps.
        rng : cupy.random.Generator
            Random number generator.
        """
        self.pre_idx = cp.array(pre_idx, dtype=cp.int32)
        self.post_idx = cp.array(post_idx, dtype=cp.int32)
        self.w = cp.array(w_init, dtype=cp.float32)
        self.E_rev = E_rev
        self.tau = tau
        self.delay_steps = delay_steps
        self.rng = rng

        # conductance values for each synapse
        self.g = cp.zeros_like(self.w)

        # exponential decay factor for synaptic conductance
        self.alpha = cp.float32(0.0)

    def set_alpha(self, alpha_val):
        """Set decay factor alpha = exp(-dt / tau)."""
        self.alpha = cp.float32(alpha_val)

    def step_decay(self):
        """Decay synaptic conductance each step."""
        self.g *= self.alpha

    def enqueue_from_pre_spikes(self, pre_spikes, scale=None):
        """
        Add synaptic conductance from active presynaptic spikes.

        Parameters
        ----------
        pre_spikes : cp.ndarray (bool)
            Spike array for all presynaptic neurons.
        scale : cp.ndarray or None
            Optional scaling factor for each connection.
        """
        active = cp.where(pre_spikes)[0]
        if active.size > 0:
            mask = cp.isin(self.pre_idx, active)
            if mask.any():
                if scale is None:
                    self.g[mask] += self.w[mask]
                else:
                    self.g[mask] += self.w[mask] * scale[mask]

    def currents_to_post(self):
        """
        Compute currents delivered to postsynaptic targets.

        Returns
        -------
        I_post : cp.ndarray
            Current array (size = max(post_idx)+1).
        post_idx : cp.ndarray
            Postsynaptic indices corresponding to synapses.
        """
        I_post = cp.zeros(int(self.post_idx.max()) + 1, dtype=cp.float32)
        cp.scatter_add(I_post, self.post_idx, self.g)
        return I_post, self.post_idx