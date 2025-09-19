import cupy as cp              # Import CuPy, a library like NumPy but it runs on the GPU
from simulate import run       # Import the main simulation runner from simulate.py

# This block ensures that the file only runs when executed directly,
# not when imported by another script.
if __name__ == "__main__":

    # --- Sanity check: make sure we are running on a machine with a CUDA GPU ---
    try:
        cp.cuda.Device(0).compute_capability   # Query compute capability of the first GPU
    except Exception as e:
        # If no GPU is found or accessible, stop the program with an error
        raise RuntimeError(
            "No CUDA GPU detected! This sim requires an NVIDIA GPU."
        ) from e

    # --- Run the simulation ---
    info = run()   # Calls the run() function from simulate.py, which executes the whole experiment

    # --- Print output location ---
    # The run() function returns a dictionary; one of its keys is "out_npz",
    # which stores the path of the saved simulation results.
    print("Saved outputs to", info["out_npz"])