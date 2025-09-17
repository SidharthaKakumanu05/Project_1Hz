import os
import numpy as np
import matplotlib.pyplot as plt

def analyze(path="cbm_py_output.npz", outdir="analysis_outputs", n_weights_to_plot=10, dt=1e-4):
    os.makedirs(outdir, exist_ok=True)
    data = np.load(path)

    # --- spikes
    io_spikes  = data["spikes/IO"]    # (T, N_CF)
    pkj_spikes = data["spikes/PKJ"]   # (T, N_PKJ)
    pf_spikes  = data["spikes/PF"]    # (T, N_PF_POOL)

    # --- weights
    t_w     = data["weights/time"]
    w_mean  = data["weights/mean"]
    w_std   = data["weights/std"]
    full_w  = data["weights/full"] if "weights/full" in data else None

    # --- CF spikes = same as IO spikes
    cf_times = np.where(io_spikes)[0] * dt

    t = np.arange(io_spikes.shape[0]) * dt

    # ======================================================
    # IO raster
    plt.figure(figsize=(10,4))
    for n in range(io_spikes.shape[1]):
        spikes = np.where(io_spikes[:,n])[0]
        plt.vlines(t[spikes], n+0.5, n+1.5, color="black")
    plt.title("IO spike raster")
    plt.xlabel("Time (s)")
    plt.ylabel("IO neuron")
    plt.savefig(os.path.join(outdir, "io_raster.png"), dpi=150)
    plt.close()

    # PKJ raster
    plt.figure(figsize=(10,4))
    for n in range(pkj_spikes.shape[1]):
        spikes = np.where(pkj_spikes[:,n])[0]
        plt.vlines(t[spikes], n+0.5, n+1.5, color="blue")
    plt.title("PKJ spike raster")
    plt.xlabel("Time (s)")
    plt.ylabel("PKJ neuron")
    plt.savefig(os.path.join(outdir, "pkj_raster.png"), dpi=150)
    plt.close()

    # PF raster (downsample because there are many PFs)
    plt.figure(figsize=(10,4))
    n_show = min(100, pf_spikes.shape[1])
    for n in range(n_show):
        spikes = np.where(pf_spikes[:,n])[0]
        plt.vlines(t[spikes], n+0.5, n+1.5, color="green")
    plt.title(f"PF spike raster (showing {n_show})")
    plt.xlabel("Time (s)")
    plt.ylabel("PF index")
    plt.savefig(os.path.join(outdir, "pf_raster.png"), dpi=150)
    plt.close()

    # PF→PKJ weights (mean/std + CF overlay)
    plt.figure(figsize=(10,4))
    plt.plot(t_w, w_mean, label="mean weight")
    plt.fill_between(t_w, w_mean-w_std, w_mean+w_std, alpha=0.3, label="±std")
    for cf_t in cf_times:
        plt.axvline(cf_t, color="red", alpha=0.05)
    plt.title("PF→PKJ weight dynamics with CF spikes")
    plt.xlabel("Time (s)")
    plt.ylabel("Weight")
    plt.legend()
    plt.savefig(os.path.join(outdir, "weights_mean_std.png"), dpi=150)
    plt.close()

    # Individual PF→PKJ weights (if full matrix available)
    if full_w is not None:
        idx = np.random.choice(full_w.shape[1], size=min(n_weights_to_plot, full_w.shape[1]), replace=False)
        plt.figure(figsize=(10,4))
        for j in idx:
            plt.plot(t_w, full_w[:,j], lw=0.8)
        plt.title(f"Sample of {len(idx)} PF→PKJ weights")
        plt.xlabel("Time (s)")
        plt.ylabel("Weight")
        plt.savefig(os.path.join(outdir, "weights_individual.png"), dpi=150)
        plt.close()

    # ======================================================
    # IO firing rate (overall)
    total_time = io_spikes.shape[0] * dt
    n_spikes_per_neuron = np.sum(io_spikes, axis=0)
    rates = n_spikes_per_neuron / total_time
    mean_rate = np.mean(rates)
    print(f"Mean IO firing rate across {io_spikes.shape[1]} neurons: {mean_rate:.3f} Hz")

    # time-binned firing rate curve
    bin_size = int(0.1 / dt)  # 100 ms bins
    n_bins = io_spikes.shape[0] // bin_size
    fr_curve = []
    for b in range(n_bins):
        start, end = b*bin_size, (b+1)*bin_size
        n_spikes = np.sum(io_spikes[start:end])
        fr_curve.append(n_spikes / (io_spikes.shape[1] * (bin_size*dt)))
    fr_curve = np.array(fr_curve)
    bin_times = np.arange(n_bins) * (bin_size*dt)

    plt.figure(figsize=(8,4))
    plt.plot(bin_times, fr_curve, marker="o")
    plt.title("IO firing rate over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.savefig(os.path.join(outdir, "io_firing_rate.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    analyze()