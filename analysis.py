import os
import numpy as np
import matplotlib.pyplot as plt


def plot_raster(spikes, dt, title, fname, outdir):
    T, N = spikes.shape
    t = np.arange(T) * dt
    fig, ax = plt.subplots(figsize=(10, 4))
    for n in range(N):
        spk_times = t[spikes[:, n] > 0]
        ax.vlines(spk_times, n, n + 0.9, color="black", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


def plot_firing_rate(spikes, dt, title, fname, outdir, bin_size=0.1):
    T, N = spikes.shape
    t = np.arange(T) * dt
    bin_steps = int(bin_size / dt)
    counts = np.add.reduceat(spikes.sum(axis=1), np.arange(0, T, bin_steps))
    rates = counts / (bin_size * N)
    t_bins = np.arange(len(rates)) * bin_size
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_bins, rates, color="red")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


def plot_weights(weights, dt, every_steps, outdir):
    n_snapshots, M = weights.shape
    t = np.arange(n_snapshots) * every_steps * dt

    # individual traces
    fig, ax = plt.subplots(figsize=(10, 4))
    n_plot = min(M, 50)
    for i in range(n_plot):
        ax.plot(t, weights[:, i], alpha=0.5, linewidth=0.5)
    ax.set_title("PF→PKJ weights (individual)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_individual.png"), dpi=200)
    plt.close(fig)

    # mean ± std
    mean = weights.mean(axis=1)
    std = weights.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, mean, label="mean", color="black")
    ax.fill_between(t, mean - std, mean + std, alpha=0.3, color="gray", label="±std")
    ax.set_title("PF→PKJ weights (mean ± std)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_mean_std.png"), dpi=200)
    plt.close(fig)


def plot_io_pair(npz_path, outdir):
    data = np.load(npz_path, allow_pickle=True)
    if "io_voltage_trace" not in data:
        print("No IO voltage trace found in this file.")
        return
    trace = data["io_voltage_trace"]
    t  = trace[:, 0]
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


def analyze(npz_path, outdir="analysis_outputs"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data = np.load(npz_path, allow_pickle=True)
    dt = 1e-4
    every_steps = int(0.01 / dt)

    for pop in ["IO", "PKJ", "PF"]:
        key = f"{pop}_spikes"
        if key in data:
            plot_raster(data[key], dt, f"{pop} raster", f"{pop.lower()}_raster.png", outdir)

    if "IO_spikes" in data:
        plot_firing_rate(data["IO_spikes"], dt, "IO firing rate", "io_firing_rate.png", outdir)

    if "PF_PKJ_weights" in data:
        plot_weights(data["PF_PKJ_weights"], dt, every_steps, outdir)

    # new: IO pair voltage trace
    plot_io_pair(npz_path, outdir)

    print(f"Analysis complete. PNGs saved to {outdir}")


if __name__ == "__main__":
    analyze("cbm_py_output.npz")