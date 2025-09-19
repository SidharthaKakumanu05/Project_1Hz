import numpy as np
import time

class Recorder:
    def __init__(self, n_steps, pop_sizes, log_stride=10, rec_weight_every=100):
        """
        Parameters
        ----------
        n_steps : int
            Total simulation steps
        pop_sizes : dict
            Mapping of population name -> number of cells
            e.g. {"PF": 4096, "PKJ": 128, "IO": 3, "BC": 32, "DCN": 8}
        log_stride : int
            Bin size (#steps per logged row)
        rec_weight_every : int
            Log weights every this many steps
        """
        self.n_steps = n_steps
        self.log_stride = log_stride
        self.rec_weight_every = rec_weight_every

        n_bins = n_steps // log_stride

        # Per-pop spike **counts** per bin (uint16 is plenty; max per bin = log_stride)
        self.spikes = {
            name: np.zeros((n_bins, size), dtype=np.uint16)
            for name, size in pop_sizes.items()
        }

        self.weights = []  # stacked at finalize
        self._start_time = None

    def start_timer(self):
        self._start_time = time.time()

    def _to_numpy(self, arr):
        # Accept CuPy or NumPy; ensure uint8 for counting
        if hasattr(arr, "get"):  # CuPy
            arr = arr.get()
        return np.asarray(arr, dtype=np.uint8)

    def log_spikes(self, step, pop_name, spikes):
        """
        Accumulate spikes into the current bin so we never miss 1-step spikes.
        """
        bin_idx = step // self.log_stride
        # Guard in case n_steps isn't perfectly divisible
        if bin_idx >= self.spikes[pop_name].shape[0]:
            return
        self.spikes[pop_name][bin_idx] += self._to_numpy(spikes)

    def maybe_log_weights(self, step, weights):
        if step % self.rec_weight_every == 0:
            if hasattr(weights, "get"):  # CuPy
                self.weights.append(weights.get())
            else:
                self.weights.append(np.asarray(weights))

    def stop_and_summary(self, steps_done):
        elapsed = time.time() - self._start_time
        steps_per_sec = steps_done / elapsed if elapsed > 0 else float("inf")
        print("\n===== Simulation Timing Summary =====")
        print(f"Total steps: {steps_done}")
        print(f"Elapsed: {elapsed:.2f} sec")
        print(f"Speed: {steps_per_sec:.2f} steps/sec")
        print("=====================================\n")
        return elapsed, steps_per_sec

    def finalize_npz(self, path):
        out = {f"{pop}_spikes": arr for pop, arr in self.spikes.items()}
        if len(self.weights) > 0:
            out["weights"] = np.stack(self.weights, axis=0)
        np.savez(path, **out)