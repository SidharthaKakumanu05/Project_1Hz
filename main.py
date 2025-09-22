#!/usr/bin/env python3
"""
Main entry point for the cerebellar microcircuit simulation.

This script serves as the starting point for running a complete simulation of a
simplified cerebellar network. The simulation models multiple neuron populations
(IO, PKJ, BC, DCN, PF, MF) with realistic connectivity and plasticity rules.

For a freshman undergrad: This is like the "main" function - it's where everything starts!
"""

import cupy as cp              # Import CuPy, a library like NumPy but it runs on the GPU
from simulate import run       # Import the main simulation runner from simulate.py

# This block ensures that the file only runs when executed directly,
# not when imported by another script.
# Think of this as: "Only run the simulation if someone double-clicks this file"
if __name__ == "__main__":

    # --- Sanity check: make sure we are running on a machine with a CUDA GPU ---
    # Why do we need this? The simulation uses GPU acceleration for speed.
    # Without a GPU, the simulation would be too slow to be useful.
    try:
        cp.cuda.Device(0).compute_capability   # Query compute capability of the first GPU
    except Exception as e:
        # If no GPU is found or accessible, stop the program with an error
        raise RuntimeError(
            "No CUDA GPU detected! This sim requires an NVIDIA GPU."
        ) from e

    # --- Run the simulation ---
    # This is where the magic happens! The run() function does everything:
    # - Sets up the network
    # - Runs the simulation for the specified time
    # - Saves all the results
    info = run()   # Calls the run() function from simulate.py, which executes the whole experiment

    # --- Print output location ---
    # The run() function returns a dictionary with useful information.
    # One of its keys is "out_npz", which stores the path of the saved simulation results.
    # This tells you where to find your data files!
    print("Saved outputs to", info["out_npz"])