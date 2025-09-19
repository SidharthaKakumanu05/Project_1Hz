import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from config import get_config


# ---------- IO raster with per-neuron frequencies ----------
def plot_io_raster_with_freq(spikes, dt, log_stride, fname, outdir):
    T, N = spikes.shape
    t = cp.arange(T) * dt * log_stride
    fig, ax = plt.subplots(figsize=(12, 6))

    spikes_np = cp.asnumpy(spikes)
    for n in range(N):
        spk_times = t[spikes_np[:, n] > 0]
        ax.vlines(cp.asnumpy(spk_times), n, n + 0.9, color="black", linewidth=0.3)

    # compute firing frequencies
    freqs = spikes_np.sum(axis=0) / (T * dt * log_stride)
    yticks = range(N)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{i} ({freqs[i]:.1f} Hz)" for i in yticks])

    ax.set_title("IO raster (with per-neuron firing rates)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


# ---------- PKJ raster ----------
def plot_pkj_raster(spikes, dt, log_stride, fname, outdir):
    T, N = spikes.shape
    t = cp.arange(T) * dt * log_stride
    fig, ax = plt.subplots(figsize=(12, 4))

    spikes_np = cp.asnumpy(spikes)
    for n in range(N):
        spk_times = t[spikes_np[:, n] > 0]
        ax.vlines(cp.asnumpy(spk_times), n, n + 0.9, color="black", linewidth=0.3)

    ax.set_title(f"PKJ raster (first {N} neurons)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


# ---------- PF raster (subsampled random neurons) ----------
def plot_pf_raster(spikes, dt, log_stride, fname, outdir):
    T, N = spikes.shape
    t = cp.arange(T) * dt * log_stride
    fig, ax = plt.subplots(figsize=(12, 4))

    spikes_np = cp.asnumpy(spikes)
    for n in range(N):
        spk_times = t[spikes_np[:, n] > 0]
        ax.vlines(cp.asnumpy(spk_times), n, n + 0.9, color="black", linewidth=0.3)

    ax.set_title(f"PF raster (random {N} neurons)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index (subset)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


# ---------- weights plotting (batch-safe) ----------
def plot_weights(weights, dt, every_steps, outdir, max_plot=50):
    n_snapshots, M = weights.shape
    t = np.arange(n_snapshots) * every_steps * dt

    # individual traces
    fig, ax = plt.subplots(figsize=(10, 4))
    n_plot = min(M, max_plot)
    for i in range(n_plot):
        ax.plot(t, weights[:, i], alpha=0.5, linewidth=0.5)
    ax.set_title("PF→PKJ weights (individual)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_individual.png"), dpi=200)
    plt.close(fig)

    # mean ± std (batch if too large)
    batch = 1000
    means = []
    stds = []
    for i in range(0, M, batch):
        chunk = weights[:, i:i + batch]
        means.append(chunk.mean(axis=1))
        stds.append(chunk.std(axis=1))

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, mean, label="mean", color="black")
    ax.fill_between(t, mean - std, mean + std,
                    alpha=0.3, color="gray", label="±std")
    ax.set_title("PF→PKJ weights (mean ± std)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_mean_std.png"), dpi=200)
    plt.close(fig)


# ---------- IO pair voltage trace ----------
def plot_io_pair(npz_path, outdir):
    data = np.load(npz_path, allow_pickle=True)
    if "io_voltage_trace" not in data:
        print("No IO voltage trace found in this file.")
        return
    trace = data["io_voltage_trace"]
    t = trace[:, 0]
    v0 = trace[:, 1]
    v1 = trace[:, 2]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, v0, label="IO neuron i0")
    ax.plot(t, v1, label="IO neuron i1")
    ax.axhline(-0.052, color="r", ls="--", label="Threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Coupled IO voltages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "io_pair_voltages.png"), dpi=200)
    plt.close(fig)


# ---------- main analysis ----------
def analyze(npz_path, outdir="analysis_outputs"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    np_data = np.load(npz_path, allow_pickle=True)
    cfg = get_config()
    dt = cfg["dt"]
    log_stride = 10  # must match Recorder
    every_steps = cfg["rec_weight_every_steps"]

    start = time.time()

    # IO
    if "IO_spikes" in np_data:
        spikes = np_data["IO_spikes"][:, :50]  # first 50 neurons
        plot_io_raster_with_freq(cp.asarray(spikes), dt, log_stride, "io_raster.png", outdir)

    # PKJ
    if "PKJ_spikes" in np_data:
        spikes = np_data["PKJ_spikes"][:, :200]  # first 200 neurons
        plot_pkj_raster(cp.asarray(spikes), dt, log_stride, "pkj_raster.png", outdir)

    # PF
    if "PF_spikes" in np_data:
        N = np_data["PF_spikes"].shape[1]
        sel = np.random.choice(N, size=min(200, N), replace=False)
        spikes = np_data["PF_spikes"][:, sel]
        plot_pf_raster(cp.asarray(spikes), dt, log_stride, "pf_raster.png", outdir)

    # Weights
    if "weights" in np_data:  # Recorder saves this as "weights"
        weights = np_data["weights"]
        plot_weights(weights, dt, every_steps, outdir)

    # IO pair trace
    plot_io_pair(npz_path, outdir)

    elapsed = time.time() - start
    print(f"Analysis complete in {elapsed:.1f} seconds. PNGs saved to {outdir}")


if __name__ == "__main__":
    analyze("cbm_py_output.npz")
