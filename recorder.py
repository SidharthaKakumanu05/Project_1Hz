import cupy as cp
import numpy as np


class Recorder:
    """
    Records spikes and weights during simulation.
    Data is stored on GPU (CuPy) until finalize_npz(),
    where it is converted back to NumPy for saving.
    """

    def __init__(self, cfg):
        self.spikes = {pop: [] for pop in ["IO", "PKJ", "BC", "DCN", "PF"]}
        self.weights = []
        self.cfg = cfg

    def log_spikes(self, pop, spikes):
        """
        Log spike vector for a population.
        Stored as CuPy arrays on GPU.
        """
        self.spikes[pop].append(spikes.astype(cp.int8))

    def maybe_log_weights(self, step, weights):
        """
        Log PF→PKJ weights at defined intervals.
        """
        if step % self.cfg["rec_weight_every_steps"] == 0:
            self.weights.append(weights.copy())

    def finalize_npz(self, path):
        """
        Convert all logged data to NumPy and save to .npz.
        """
        out = {}

        # Spikes: list of timesteps → stack into [T, N]
        for pop, seq in self.spikes.items():
            if len(seq) > 0:
                out[f"{pop}_spikes"] = np.stack([cp.asnumpy(x) for x in seq])
            else:
                out[f"{pop}_spikes"] = np.zeros((0,), dtype=np.int8)

        # Weights: [timepoints, num_synapses]
        if len(self.weights) > 0:
            out["PF_PKJ_weights"] = np.stack([cp.asnumpy(w) for w in self.weights])
        else:
            out["PF_PKJ_weights"] = np.zeros((0,), dtype=np.float32)

        np.savez(path, **out)