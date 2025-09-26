import cupy as cp
from simulate import run

if __name__ == "__main__":
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as e:
        raise RuntimeError(
            "No CUDA GPU detected! This sim requires an NVIDIA GPU."
        ) from e

    info = run()
    print("Saved outputs to", info["out_npz"])