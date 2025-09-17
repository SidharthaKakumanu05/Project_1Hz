import cupy as cp


class SynapseProj:
    """
    Generic synapse projection:
    - manages delay lines
    - computes decaying conductances
    - handles pre→post enqueue
    """

    def __init__(self, pre_idx, post_idx, w_init, E_rev, tau, delay_steps):
        self.pre_idx = cp.asarray(pre_idx, dtype=cp.int32)
        self.post_idx = cp.asarray(post_idx, dtype=cp.int32)

        self.M = self.pre_idx.size
        self.w = cp.asarray(w_init, dtype=cp.float32)

        self.E_rev = cp.float32(E_rev)
        self.tau = tau
        self.delay_steps = delay_steps

        # delay buffer
        self.delay_buf = cp.zeros((delay_steps + 1, self.M), dtype=cp.float32)
        self.buf_ptr = 0

        # decay factor (set later by set_alpha)
        self.alpha = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def enqueue_from_pre_spikes(self, pre_spikes, scale=None):
        """
        Insert spikes into delay buffer, scaled by weights (and optional scale factors).
        """
        active = pre_spikes[self.pre_idx].astype(cp.float32)
        if scale is not None:
            active = active * scale
        contrib = active * self.w
        self.delay_buf[self.buf_ptr, :] += contrib

    def step_decay(self):
        """
        Advance buffer pointer and decay conductances.
        """
        self.buf_ptr = (self.buf_ptr + 1) % self.delay_buf.shape[0]
        self.delay_buf[self.buf_ptr, :] *= 0.0  # reset slot

    def currents_to_post(self):
        """
        Sum active conductances, apply decay, return (g, post_idx).
        """
        # effective conductance = sum over delay slots
        g = cp.sum(self.delay_buf, axis=0)
        return g, self.post_idx