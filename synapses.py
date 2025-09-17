
import numpy as np

class SynapseProj:
    __slots__ = ("pre_idx","post_idx","w","E_rev","g","delay_buf","delay_head","tau","alpha","max_delay","M")
    def __init__(self, pre_idx, post_idx, w_init, E_rev, tau, delay_steps, rng):
        self.pre_idx = pre_idx.astype(np.int32)
        self.post_idx= post_idx.astype(np.int32)
        self.w = np.array(w_init, dtype=np.float32)
        self.E_rev = np.float32(E_rev)
        self.g = np.zeros_like(self.w, dtype=np.float32)
        self.max_delay = int(delay_steps)
        self.delay_buf = np.zeros((self.w.size, self.max_delay+1), dtype=np.float32)
        self.delay_head = 0
        self.tau = np.float32(tau)
        self.alpha = None
        self.M = self.w.size

    def set_alpha(self, alpha):
        self.alpha = np.float32(alpha)

    def step_decay(self):
        self.g *= self.alpha
        self.delay_head = (self.delay_head + 1) % (self.max_delay+1)
        self.g += self.delay_buf[:, self.delay_head]
        self.delay_buf[:, self.delay_head] = 0.0

    def enqueue_from_pre_spikes(self, pre_spike_mask, scale=None):
        if not np.any(pre_spike_mask):
            return
        con_mask = pre_spike_mask[self.pre_idx]
        if not np.any(con_mask):
            return
        inc = self.w.copy()
        if scale is not None:
            inc *= scale
        slot = (self.delay_head + self.max_delay) % (self.max_delay+1)
        self.delay_buf[con_mask, slot] += inc[con_mask]

    def currents_to_post(self):
        return self.g, self.post_idx
