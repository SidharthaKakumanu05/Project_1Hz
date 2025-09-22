#!/usr/bin/env python3
"""
Data recording and storage for the cerebellar microcircuit simulation.

This file implements the Recorder class, which handles logging of all simulation
data including spikes, synaptic weights, and timing information. The recorder
efficiently stores data in a compressed format for later analysis.

For a freshman undergrad: This is like the "data logger" - it keeps track of
everything that happens during the simulation so we can analyze it later!
"""

import numpy as np
import time

class Recorder:
    """
    Recorder class for logging simulation data.
    
    This class efficiently stores spikes and synaptic weights during the simulation.
    It uses binned spike counting to save memory while preserving temporal resolution.
    """
    
    def __init__(self, n_steps, pop_sizes, log_stride=10, rec_weight_every=100):
        """
        Initialize the recorder with simulation parameters.

        Parameters
        ----------
        n_steps : int
            Total number of simulation steps (determines how much data to store)
        pop_sizes : dict
            Mapping of population name -> number of cells
            Example: {"PF": 4096, "PKJ": 128, "IO": 16, "BC": 512, "DCN": 8}
        log_stride : int
            How many steps are grouped together per log bin (for spikes).
            Larger values = less memory usage but lower temporal resolution
        rec_weight_every : int
            Record synaptic weights every this many steps.
            More frequent recording = more detailed weight evolution data
        """
        self.n_steps = n_steps
        self.log_stride = log_stride
        self.rec_weight_every = rec_weight_every

        n_bins = n_steps // log_stride   # total rows in the spike logs

        # Dictionary of spike logs for each population.
        # Each entry is a 2D array of shape (n_bins, num_cells).
        # Each element stores the *count* of spikes a neuron fired within that bin.
        # Using uint16 allows up to 65,535 spikes per bin (more than enough!)
        self.spikes = {
            name: np.zeros((n_bins, size), dtype=np.uint16)
            for name, size in pop_sizes.items()
        }

        # Weight snapshots (list of arrays collected over time).
        # Each entry is a snapshot of all synaptic weights at one time point
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
        
        Handles both CuPy (GPU) and NumPy arrays by converting GPU arrays to CPU.
        This is necessary because we need to store data on the CPU for saving to disk.
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
        
        This function efficiently stores spikes by binning them in time.
        Instead of storing every single spike, we count how many spikes
        each neuron fired in each time bin. This saves memory while
        preserving the essential information about firing patterns.
        
        Parameters
        ----------
        step : int
            Current simulation step
        pop_name : str
            Name of the neuron population (e.g., "PF", "PKJ", "IO")
        spikes : array
            Boolean array indicating which neurons spiked this step
        """
        bin_idx = step // self.log_stride
        # Guard: in case steps don't divide evenly into bins, ignore out-of-range
        if bin_idx >= self.spikes[pop_name].shape[0]:
            return
        # Add to bin: note this adds elementwise across neurons
        # Convert to numpy only when necessary (GPU → CPU)
        if hasattr(spikes, "get"):
            spikes_np = spikes.get().astype(np.uint8)
        else:
            spikes_np = np.asarray(spikes, dtype=np.uint8)
        self.spikes[pop_name][bin_idx] += spikes_np

    # -------------------------
    # Weight logging
    # -------------------------
    def maybe_log_weights(self, step, weights):
        """
        Occasionally log synaptic weights (every rec_weight_every steps).
        
        Synaptic weights change slowly compared to spikes, so we don't need
        to record them every time step. Recording every few steps gives us
        enough temporal resolution to see how learning progresses.
        
        Parameters
        ----------
        step : int
            Current simulation step
        weights : array
            Current synaptic weights (usually PF→PKJ weights)
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
        
        This gives us performance metrics to see how fast the simulation ran.
        
        Parameters
        ----------
        steps_done : int
            Number of simulation steps that were completed
            
        Returns
        -------
        elapsed : float
            Total elapsed time in seconds
        steps_per_sec : float
            Simulation speed in steps per second
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
        
        This creates a compressed archive containing all the simulation data.
        The .npz format is efficient for storing multiple NumPy arrays and
        is the standard format for scientific data in Python.
        
        Data stored:
        - Spikes: stored as <pop_name>_spikes arrays (e.g., "PF_spikes", "PKJ_spikes")
        - Weights: stacked along time if recorded (shows how weights changed over time)
        
        Parameters
        ----------
        path : str
            File path where to save the .npz file
        """
        out = {f"{pop}_spikes": arr for pop, arr in self.spikes.items()}

        if len(self.weights) > 0:
            out["weights"] = np.stack(self.weights, axis=0)  # Stack along time axis

        np.savez(path, **out)   # compressed archive of arrays