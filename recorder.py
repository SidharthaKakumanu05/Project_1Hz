import numpy as np
import cupy as cp


class Recorder:
    def __init__(self, cfg):
        self.dt = cfg["dt"]
        self.rec_weight_every_steps = cfg["rec_weight_every_steps"]
        self.buffers = {
            "IO_spikes": [],
            "PKJ_spikes": [],
            "BC_spikes": [],
            "DCN_spikes": [],
            "PF_spikes": [],
            "PF_PKJ_weights": [],
        }

    def log_spikes(self, pop, spikes):
        self.buffers[f"{pop}_spikes"].append(cp.asnumpy(spikes.astype(np.int8)))

    def maybe_log_weights(self, step, weights):
        if step % self.rec_weight_every_steps == 0:
            self.buffers["PF_PKJ_weights"].append(cp.asnumpy(weights))

    def finalize_npz(self, path):
        out = {}
        for k, v in self.buffers.items():
            if len(v) > 0:
                out[k] = np.stack(v, axis=0)
        np.savez(path, **out)