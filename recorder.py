import numpy as np

class Recorder:
    def __init__(self, cfg):
        self.dt = cfg["dt"]
        self.spikes = {}
        self.series = {}
        self.weight_times = []
        self.weight_means = []
        self.weight_stds  = []
        self.weight_full  = []   # NEW
        self.weight_every = cfg["rec_weight_every_steps"]

    def log_spikes(self, name, step_mask):
        self.spikes.setdefault(name, []).append(step_mask.copy())

    def log_scalar(self, name, value):
        self.series.setdefault(name, []).append(float(value))

    def maybe_log_weights(self, step, w):
        if step % self.weight_every == 0:
            self.weight_times.append(step*self.dt)
            self.weight_means.append(float(np.mean(w)))
            self.weight_stds.append(float(np.std(w)))
            self.weight_full.append(w.copy())  # NEW: store full vector

    def finalize_npz(self, path):
        out = {}
        for k, v in self.spikes.items():
            out[f"spikes/{k}"] = np.array(v, dtype=bool)
        for k, v in self.series.items():
            out[f"series/{k}"] = np.array(v, dtype=np.float32)
        out["weights/time"] = np.array(self.weight_times, dtype=np.float32)
        out["weights/mean"] = np.array(self.weight_means, dtype=np.float32)
        out["weights/std"]  = np.array(self.weight_stds, dtype=np.float32)
        if self.weight_full:  # save full weights if logged
            out["weights/full"] = np.array(self.weight_full, dtype=np.float32)
        np.savez(path, **out)