import numpy as np
import time

class Recorder:
    def __init__(self, n_steps, pop_sizes, log_stride=10, rec_weight_every=100):
        """
        Recorder handles logging of spikes and (optionally) synaptic weights.

        Parameters
        ----------
        n_steps : int
            Total number of simulation steps.
        pop_sizes : dict
            Mapping of population name -> number of cells
            Example: {"PF": 4096, "PKJ": 128, "IO": 3, "BC": 32, "DCN": 8}
        log_stride : int
            How many steps are grouped together per log bin (for spikes).
        rec_weight_every : int
            Record synaptic weights every this many steps.
        """
        self.n_steps = n_steps
        self.log_stride = log_stride
        self.rec_weight_every = rec_weight_every

        n_bins = n_steps // log_stride   # total rows in the spike logs

        # Dictionary of spike logs for each population.
        # Each entry is a 2D array of shape (n_bins, num_cells).
        # Each element stores the *count* of spikes a neuron fired within that bin.
        self.spikes = {
            name: np.zeros((n_bins, size), dtype=np.uint16)
            for name, size in pop_sizes.items()
        }

        # Weight snapshots (list of arrays collected over time).
        self.weights = []

        # Timing marker (set when sim starts).
        self._start_time = None

    # -------------------------
    # Timer utilities
    # -------------------------
    def start_timer(self):
        """Mark the start time of the simulation."""
        self._start_time = time.time()

    def _to_numpy(self, arr):
        """
        Convert input to a NumPy array of uint8.
        Handles both CuPy (GPU) and NumPy arrays.
        """
        if hasattr(arr, "get"):  # CuPy arrays have a .get() method to copy to CPU
            arr = arr.get()
        return np.asarray(arr, dtype=np.uint8)

    # -------------------------
    # Spike logging
    # -------------------------
    def log_spikes(self, step, pop_name, spikes):
        """
        Add spikes from one population at this time step to the correct bin.
        This ensures 1-step spikes aren't missed when compressing into bins.
        """
        bin_idx = step // self.log_stride
        # Guard: in case steps don't divide evenly into bins, ignore out-of-range
        if bin_idx >= self.spikes[pop_name].shape[0]:
            return
        # Add to bin: note this adds elementwise across neurons
        self.spikes[pop_name][bin_idx] += self._to_numpy(spikes)

    # -------------------------
    # Weight logging
    # -------------------------
    def maybe_log_weights(self, step, weights):
        """
        Occasionally log synaptic weights (every rec_weight_every steps).
        """
        if step % self.rec_weight_every == 0:
            if hasattr(weights, "get"):    # handle CuPy → NumPy
                self.weights.append(weights.get())
            else:
                self.weights.append(np.asarray(weights))

    # -------------------------
    # Timing summary
    # -------------------------
    def stop_and_summary(self, steps_done):
        """
        Stop timer, print timing stats, and return elapsed time + speed.
        """
        elapsed = time.time() - self._start_time
        steps_per_sec = steps_done / elapsed if elapsed > 0 else float("inf")

        print("\n===== Simulation Timing Summary =====")
        print(f"Total steps: {steps_done}")
        print(f"Elapsed: {elapsed:.2f} sec")
        print(f"Speed: {steps_per_sec:.2f} steps/sec")
        print("=====================================\n")

        return elapsed, steps_per_sec

    # -------------------------
    # Finalize: save outputs
    # -------------------------
    def finalize_npz(self, path):
        """
        Save all recorded data to a .npz file.

        - Spikes: stored as <pop_name>_spikes arrays
        - Weights: stacked along time if recorded
        """
        out = {f"{pop}_spikes": arr for pop, arr in self.spikes.items()}

        if len(self.weights) > 0:
            out["weights"] = np.stack(self.weights, axis=0)

        np.savez(path, **out)   # compressed archive of arrays