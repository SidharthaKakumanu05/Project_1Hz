#!/usr/bin/env python3
"""
Data analysis and visualization for the cerebellar microcircuit simulation.

This file loads simulation results from .npz files and generates comprehensive
visualizations including raster plots, firing rate analysis, synaptic weight
evolution, and network activity patterns. It's optimized for both speed and
visual clarity to handle large datasets efficiently.

For a freshman undergrad: This is where you can see what happened during the
simulation! It creates all the graphs and plots that help you understand
how the cerebellar network behaved.
"""

import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import warnings
from config import get_config

# Suppress numba TBB warnings (these are just version compatibility warnings)
warnings.filterwarnings("ignore", message="The TBB threading layer requires TBB version")
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

# Try to import numba for performance optimization
# Numba is a JIT compiler that makes numerical computations much faster
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("Using numba for fast data processing")
except ImportError:
    print("Warning: numba not available, using slower fallback functions")
    NUMBA_AVAILABLE = False
    # Define dummy decorators so the code still works without numba
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Set matplotlib parameters for better visibility and publication-quality plots
plt.rcParams['figure.dpi'] = 300        # High resolution plots
plt.rcParams['savefig.dpi'] = 300       # High resolution saved images
plt.rcParams['font.size'] = 10          # Readable font size
plt.rcParams['axes.linewidth'] = 1.2    # Thicker axes lines
plt.rcParams['lines.linewidth'] = 1.5   # Thicker plot lines

# Optimized functions for faster data processing
# These functions use numba JIT compilation for significant speed improvements
@jit(nopython=True, parallel=True)
def compute_spike_times_numba(spikes_np, t_np):
    """
    Fast computation of spike times using numba JIT compilation.
    
    This function converts binned spike data back into individual spike times
    and neuron IDs for creating raster plots. It's optimized with numba for
    handling large datasets efficiently.
    
    Parameters
    ----------
    spikes_np : array
        2D array of shape (time_bins, neurons) containing spike counts per bin
    t_np : array
        1D array of time points corresponding to each time bin
        
    Returns
    -------
    spike_times : array
        1D array of spike times (in seconds)
    neuron_ids : array
        1D array of neuron IDs corresponding to each spike time
    """
    T, N = spikes_np.shape  # T = number of time bins, N = number of neurons
    
    # First pass: count total spikes to pre-allocate arrays efficiently
    total_spikes = 0
    for n in prange(N):  # Parallel loop over neurons
        for t_idx in range(T):
            if spikes_np[t_idx, n] > 0:
                total_spikes += 1
    
    # Pre-allocate arrays for efficiency
    spike_times = np.zeros(total_spikes)
    neuron_ids = np.zeros(total_spikes, dtype=np.int32)
    
    # Second pass: fill arrays
    idx = 0
    for n in prange(N):
        for t_idx in range(T):
            if spikes_np[t_idx, n] > 0:
                spike_times[idx] = t_np[t_idx]
                neuron_ids[idx] = n
                idx += 1
    
    return spike_times, neuron_ids

@jit(nopython=True, parallel=True)
def compute_mean_rates_numba(spikes_np, window_steps, dt_log_stride):
    """Fast computation of mean firing rates using numba."""
    T, N = spikes_np.shape
    n_windows = T - window_steps
    
    # Pre-allocate arrays
    mean_rates = np.zeros(n_windows)
    time_points = np.zeros(n_windows)
    
    for i in prange(n_windows):
        window_sum = 0.0
        start_idx = i
        end_idx = i + window_steps
        
        for t_idx in range(start_idx, end_idx):
            for n in range(N):
                window_sum += spikes_np[t_idx, n]
        
        window_rate = window_sum / (window_steps * N * dt_log_stride)
        mean_rates[i] = window_rate
        time_points[i] = (i + window_steps) * dt_log_stride
    
    return mean_rates, time_points

# Fallback functions when numba is not available
def compute_spike_times_fallback(spikes_np, t_np):
    """Fallback function for spike times computation without numba."""
    T, N = spikes_np.shape
    spike_times = []
    neuron_ids = []
    
    for n in range(N):
        spk_times = t_np[spikes_np[:, n] > 0]
        if len(spk_times) > 0:
            spike_times.extend(spk_times)
            neuron_ids.extend([n] * len(spk_times))
    
    return np.array(spike_times), np.array(neuron_ids)

def compute_mean_rates_fallback(spikes_np, window_steps, dt_log_stride):
    """Fallback function for mean rates computation without numba."""
    T, N = spikes_np.shape
    mean_rates = []
    time_points = []
    
    for i in range(window_steps, T):
        window_spikes = spikes_np[i-window_steps:i, :]
        window_rate = np.mean(window_spikes) / dt_log_stride
        mean_rates.append(window_rate)
        time_points.append(i * dt_log_stride)
    
    return np.array(mean_rates), np.array(time_points)


# --------------------------------------------
# IO raster plot with per-neuron frequencies
# --------------------------------------------
def plot_io_raster_with_freq(spikes, dt, log_stride, fname, outdir):
    """
    Plot a raster of IO spikes and label each neuron with its firing rate.
    """
    T, N = spikes.shape
    t_np = np.arange(T) * dt * log_stride   # time vector (seconds)

    # Scale figure size based on number of neurons and time duration (with reasonable limits)
    raw_width = T * dt * log_stride * 0.1
    fig_width = min(max(12, raw_width), 100)  # Max 100 inches width
    fig_height = max(6, N * 0.3)  # 0.3 inches per neuron
    
    if raw_width > 100:
        print(f"Warning: IO raster figure width limited to 100 inches (requested: {raw_width:.1f})")
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Convert to numpy once
    if hasattr(spikes, 'get'):
        spikes_np = spikes.get()
    else:
        spikes_np = np.asarray(spikes)

    # Use vlines approach with downsampling for very long simulations
    if raw_width > 100:
        # For very long simulations, downsample time points to keep all neurons visible
        downsample_factor = max(1, int(raw_width / 80))  # Downsample to fit in 80 inches
        print(f"Downsampling IO raster by factor {downsample_factor} to show all data")
        t_downsampled = t_np[::downsample_factor]
        spikes_downsampled = spikes_np[::downsample_factor, :]
        
        # Plot using vlines with downsampled data
        for n in range(N):
            spk_times = t_downsampled[spikes_downsampled[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="black", linewidth=0.5, alpha=0.8)
    else:
        # For shorter simulations, use vlines with all data
        for n in range(N):
            spk_times = t_np[spikes_np[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="black", linewidth=0.5, alpha=0.8)

    # Compute average firing frequency per neuron (vectorized)
    freqs = spikes_np.sum(axis=0) / (T * dt * log_stride)
    yticks = range(N)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{i} ({freqs[i]:.3f} Hz)" for i in yticks])

    ax.set_title("IO raster (with per-neuron firing rates)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Neuron index (Hz)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# PKJ raster plot
# --------------------------------------------
def plot_pkj_raster(spikes, dt, log_stride, fname, outdir):
    """
    Plot a raster of PKJ spikes (subset of neurons).
    """
    T, N = spikes.shape
    t_np = np.arange(T) * dt * log_stride

    # Scale figure size based on data (with reasonable limits)
    fig_width = min(max(12, T * dt * log_stride * 0.1), 100)  # Max 100 inches width
    fig_height = max(4, N * 0.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Convert to numpy once
    if hasattr(spikes, 'get'):
        spikes_np = spikes.get()
    else:
        spikes_np = np.asarray(spikes)

    # Use vlines approach with downsampling for very long simulations
    raw_width = T * dt * log_stride * 0.1
    if raw_width > 100:
        # For very long simulations, downsample time points to keep all neurons visible
        downsample_factor = max(1, int(raw_width / 80))
        print(f"Downsampling PKJ raster by factor {downsample_factor} to show all data")
        t_downsampled = t_np[::downsample_factor]
        spikes_downsampled = spikes_np[::downsample_factor, :]
        
        # Plot using vlines with downsampled data
        for n in range(N):
            spk_times = t_downsampled[spikes_downsampled[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="blue", linewidth=0.5, alpha=0.9)
    else:
        # For shorter simulations, use vlines with all data
        for n in range(N):
            spk_times = t_np[spikes_np[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="blue", linewidth=0.5, alpha=0.9)

    ax.set_title(f"PKJ raster (first {N} neurons)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Neuron index", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# PF raster plot (subset)
# --------------------------------------------
def plot_pf_raster(spikes, dt, log_stride, fname, outdir):
    """
    Plot a raster of PF spikes for a subset of neurons (randomly selected).
    """
    T, N = spikes.shape
    t_np = np.arange(T) * dt * log_stride

    # Scale figure size based on data (with reasonable limits)
    fig_width = min(max(12, T * dt * log_stride * 0.1), 100)  # Max 100 inches width
    fig_height = max(4, N * 0.15)  # Smaller height per neuron for PF since there are many
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Convert to numpy once
    if hasattr(spikes, 'get'):
        spikes_np = spikes.get()
    else:
        spikes_np = np.asarray(spikes)

    # Use vlines approach with downsampling for very long simulations
    raw_width = T * dt * log_stride * 0.1
    if raw_width > 100:
        # For very long simulations, downsample time points to keep all neurons visible
        downsample_factor = max(1, int(raw_width / 80))
        print(f"Downsampling PF raster by factor {downsample_factor} to show all data")
        t_downsampled = t_np[::downsample_factor]
        spikes_downsampled = spikes_np[::downsample_factor, :]
        
        # Plot using vlines with downsampled data
        for n in range(N):
            spk_times = t_downsampled[spikes_downsampled[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="green", linewidth=0.3, alpha=0.7)
    else:
        # For shorter simulations, use vlines with all data
        for n in range(N):
            spk_times = t_np[spikes_np[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="green", linewidth=0.3, alpha=0.7)

    ax.set_title(f"PF raster (random {N} neurons)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Neuron index (subset)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# DCN raster plot with per-neuron frequencies
# --------------------------------------------
def plot_dcn_raster_with_freq(spikes, dt, log_stride, fname, outdir):
    """
    Plot a raster of DCN spikes and label each neuron with its firing rate.
    """
    T, N = spikes.shape
    t_np = np.arange(T) * dt * log_stride   # time vector (seconds)

    # Scale figure size based on data (with reasonable limits)
    fig_width = min(max(12, T * dt * log_stride * 0.1), 100)  # Max 100 inches width
    fig_height = max(4, N * 0.4)  # More space per neuron for DCN since there are few
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Convert to numpy once
    if hasattr(spikes, 'get'):
        spikes_np = spikes.get()
    else:
        spikes_np = np.asarray(spikes)

    # Use vlines approach with downsampling for very long simulations
    raw_width = T * dt * log_stride * 0.1
    if raw_width > 100:
        # For very long simulations, downsample time points to keep all neurons visible
        downsample_factor = max(1, int(raw_width / 80))
        print(f"Downsampling DCN raster by factor {downsample_factor} to show all data")
        t_downsampled = t_np[::downsample_factor]
        spikes_downsampled = spikes_np[::downsample_factor, :]
        
        # Plot using vlines with downsampled data
        for n in range(N):
            spk_times = t_downsampled[spikes_downsampled[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="red", linewidth=0.8, alpha=0.9)
    else:
        # For shorter simulations, use vlines with all data
        for n in range(N):
            spk_times = t_np[spikes_np[:, n] > 0]
            if len(spk_times) > 0:
                ax.vlines(spk_times, n, n + 0.9, color="red", linewidth=0.8, alpha=0.9)

    # Compute average firing frequency per neuron (vectorized)
    freqs = spikes_np.sum(axis=0) / (T * dt * log_stride)
    yticks = range(N)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{i} ({freqs[i]:.3f} Hz)" for i in yticks])

    ax.set_title("DCN raster (with per-neuron firing rates)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Neuron index (Hz)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# PKJ mean firing rate over time
# --------------------------------------------
def plot_pkj_mean_rate(spikes, dt, log_stride, fname, outdir, window_size=100):
    """
    Plot the mean firing rate of PKJ neurons over time using a sliding window.
    """
    T, N = spikes.shape
    
    # Convert to numpy once
    if hasattr(spikes, 'get'):
        spikes_np = spikes.get()
    else:
        spikes_np = np.asarray(spikes)
    
    # Calculate mean firing rate using optimized function
    window_steps = window_size
    if NUMBA_AVAILABLE:
        mean_rates, time_points = compute_mean_rates_numba(spikes_np, window_steps, dt * log_stride)
    else:
        mean_rates, time_points = compute_mean_rates_fallback(spikes_np, window_steps, dt * log_stride)
    
    # Scale figure width based on time duration (with reasonable limits)
    fig_width = min(max(10, T * dt * log_stride * 0.08), 80)  # Max 80 inches width
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    
    # Use thicker line and higher contrast
    ax.plot(time_points, mean_rates, color='blue', linewidth=3, alpha=0.8)
    
    # Add shaded area to show data density
    ax.fill_between(time_points, mean_rates, alpha=0.2, color='blue')
    
    ax.set_title("PKJ mean firing rate over time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mean firing rate (Hz)", fontsize=12)
    ax.grid(True, alpha=0.4, linewidth=1)
    
    # Improve y-axis scaling
    if len(mean_rates) > 0:
        y_min, y_max = mean_rates.min(), mean_rates.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# PF→PKJ weights plotting
# --------------------------------------------
def plot_weights(weights, dt, every_steps, outdir, max_plot=50):
    """
    Plot synaptic weights over time:
    - individual traces for a subset of synapses
    - mean ± std envelope across all synapses
    """
    n_snapshots, M = weights.shape
    t = np.arange(n_snapshots) * every_steps * dt

    # Calculate weight statistics for better visualization
    weight_min = weights.min()
    weight_max = weights.max()
    weight_mean = weights.mean()
    weight_std = weights.std()
    
    print(f"Weight statistics: min={weight_min:.6f}, max={weight_max:.6f}, mean={weight_mean:.6f}, std={weight_std:.6f}")

    # ---- Individual traces ----
    # Scale figure size based on data (with reasonable limits)
    fig_width = min(max(12, len(t) * 0.05), 80)  # Max 80 inches width
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    n_plot = min(M, max_plot)
    
    # Use different colors for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_plot)))
    
    # Plot individual weight traces with better visibility
    for i in range(n_plot):
        color = colors[i % len(colors)]
        ax.plot(t, weights[:, i], alpha=0.8, linewidth=2.5, 
                color=color, marker='o', markersize=3, markevery=max(1, len(t)//50))
    
    # Set y-axis to show full range with some padding
    weight_range = weight_max - weight_min
    if weight_range > 0:
        padding = weight_range * 0.1
        ax.set_ylim(weight_min - padding, weight_max + padding)
    else:
        # If no variation, show a small range around the mean
        ax.set_ylim(weight_mean - 0.01, weight_mean + 0.01)
    
    # Add horizontal lines for reference
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Initial weight (1.0)')
    ax.axhline(y=weight_mean, color='orange', linestyle=':', linewidth=2, alpha=0.8, label=f'Mean weight ({weight_mean:.4f})')
    
    ax.set_title(f"PF→PKJ weights (individual traces, n={n_plot})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_individual.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- Mean ± std envelope (optimized) ----
    # Use vectorized operations for better performance
    if M > 10000:  # For large datasets, use chunked processing
        batch = 5000  # Larger batches for better performance
        means, stds = [], []
        for i in range(0, M, batch):
            chunk = weights[:, i:i + batch]
            means.append(chunk.mean(axis=1))
            stds.append(chunk.std(axis=1))
        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)
    else:  # For smaller datasets, use direct computation
        mean = weights.mean(axis=1)
        std = weights.std(axis=1)

    # Scale figure size based on data (with reasonable limits)
    fig_width = min(max(10, len(t) * 0.05), 80)  # Max 80 inches width
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Plot mean with thicker line
    ax.plot(t, mean, label="Mean", color="black", linewidth=4, alpha=0.9)
    
    # Fill between with better visibility
    ax.fill_between(t, mean - std, mean + std,
                    alpha=0.4, color="blue", label="±1 std")
    
    # Add additional reference lines
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Initial weight (1.0)')
    
    # Show min/max bounds
    min_val, max_val = mean.min(), mean.max()
    ax.axhline(y=min_val, color='green', linestyle=':', linewidth=2, alpha=0.6, label=f'Min ({min_val:.4f})')
    ax.axhline(y=max_val, color='orange', linestyle=':', linewidth=2, alpha=0.6, label=f'Max ({max_val:.4f})')
    
    # Set y-axis to show full range with padding
    data_min = (mean - std).min()
    data_max = (mean + std).max()
    data_range = data_max - data_min
    if data_range > 0:
        padding = data_range * 0.1
        ax.set_ylim(data_min - padding, data_max + padding)
    else:
        ax.set_ylim(mean.mean() - 0.01, mean.mean() + 0.01)
    
    ax.set_title("PF→PKJ weights (mean ± std across all synapses)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.4, linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_mean_std.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # ---- Additional plot: Weight distribution over time ----
    fig_width = min(max(10, len(t) * 0.05), 80)  # Max 80 inches width
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 10))
    
    # Plot weight histogram at different time points with better visibility
    time_indices = [0, n_snapshots//4, n_snapshots//2, 3*n_snapshots//4, n_snapshots-1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    labels = ['t=0', f't={t[n_snapshots//4]:.1f}s', f't={t[n_snapshots//2]:.1f}s', 
              f't={t[3*n_snapshots//4]:.1f}s', f't={t[-1]:.1f}s']
    
    for i, (idx, color, label) in enumerate(zip(time_indices, colors, labels)):
        ax1.hist(weights[idx, :], bins=50, alpha=0.7, color=color, label=label, density=True, 
                linewidth=1, edgecolor='black')
    
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=3, alpha=0.8, label='Initial weight (1.0)')
    ax1.axvline(x=weight_mean, color='red', linestyle=':', linewidth=2, alpha=0.8, label=f'Overall mean ({weight_mean:.4f})')
    ax1.set_xlabel("Weight", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Weight distribution at different time points", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.4, linewidth=1)
    
    # Plot percentage of weights that changed from initial value (vectorized)
    # Use vectorized operations for better performance
    tolerance = 1e-6
    unchanged_percent = 100 * np.sum(np.abs(weights - 1.0) < tolerance, axis=1) / M
    
    ax2.plot(t, unchanged_percent, color='blue', linewidth=3, alpha=0.8)
    ax2.fill_between(t, unchanged_percent, alpha=0.2, color='blue')
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Percentage of weights at 1.0", fontsize=12)
    ax2.set_title("Percentage of synapses with unchanged weights", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.4, linewidth=1)
    ax2.set_ylim(0, 100)
    
    # Add horizontal reference lines
    ax2.axhline(y=50, color='red', linestyle=':', linewidth=2, alpha=0.6, label='50% line')
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# IO pair voltage traces
# --------------------------------------------
def plot_io_pair(npz_path, outdir):
    """
    Plot the voltage traces of two IO neurons (if available).
    
    This function shows the membrane voltage evolution of IO neurons over time,
    which helps visualize how the gap junction coupling affects their synchronization.
    The plot shows both individual neuron voltages and the spike threshold.
    
    Parameters
    ----------
    npz_path : str
        Path to the simulation data file
    outdir : str
        Directory to save the plot
    """
    data = np.load(npz_path, allow_pickle=True)
    if "io_voltage_trace" not in data:
        print("No IO voltage trace found in this file.")
        return

    # Extract voltage traces for two IO neurons
    trace = data["io_voltage_trace"]
    t = trace[:, 0]    # Time points
    v0 = trace[:, 1]   # Voltage of first IO neuron
    v1 = trace[:, 2]   # Voltage of second IO neuron

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, v0, label="IO neuron 0", linewidth=2)
    ax.plot(t, v1, label="IO neuron 1", linewidth=2)
    ax.axhline(-0.052, color="r", ls="--", label="Spike Threshold", linewidth=2)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.set_title("Coupled IO Neuron Voltages", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "io_pair_voltages.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------
# Main analysis driver
# --------------------------------------------
def analyze(npz_path, outdir="analysis_outputs"):
    """
    Main analysis function that loads simulation outputs and generates comprehensive plots.
    
    This function is the main entry point for analyzing simulation results. It loads
    the data file, processes different types of data (spikes, weights, voltages),
    and generates publication-ready plots showing network behavior.
    
    Generated plots include:
    - IO raster with firing frequencies (shows teaching signal patterns)
    - PKJ raster (shows main computational units)
    - PKJ mean firing rate over time (shows network dynamics)
    - PF raster subset (shows input patterns)
    - DCN raster with firing frequencies (shows output patterns)
    - Weight evolution plots (shows learning progress)
    - IO voltage traces (shows gap junction effects)
    
    Parameters
    ----------
    npz_path : str
        Path to the simulation data file (.npz format)
    outdir : str, optional
        Directory to save analysis plots (default: "analysis_outputs")
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created output directory: {outdir}")

    print(f"Loading data from {npz_path}...")
    start = time.time()
    
    # Load data with memory mapping for large files (efficient for big datasets)
    # mmap_mode='r' loads the file on-demand rather than all at once
    np_data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    cfg = get_config()
    dt = cfg["dt"]                    # Time step size
    log_stride = 50                   # Must match Recorder log_stride
    every_steps = cfg["rec_weight_every_steps"]  # How often weights were recorded

    print(f"Data loaded in {time.time() - start:.1f} seconds")
    
    # Check if we need to downsample for very long simulations
    total_time = cfg["T_sec"]
    if total_time > 600:  # More than 10 minutes
        print(f"Warning: Very long simulation detected ({total_time:.0f}s). Some plots may be downsampled for performance.")
    
    analysis_start = time.time()

    # Process different types of data and generate plots
    print("\n=== Generating Analysis Plots ===")
    
    # IO spikes - Inferior Olive neurons (teaching signals)
    if "IO_spikes" in np_data:
        print("Processing IO spikes...")
        spikes = np_data["IO_spikes"][:, :50]  # Plot first 50 neurons for visibility
        plot_io_raster_with_freq(spikes, dt, log_stride, "io_raster.png", outdir)

    # PKJ spikes - Purkinje cells (main computational units)
    if "PKJ_spikes" in np_data:
        print("Processing PKJ spikes...")
        spikes = np_data["PKJ_spikes"][:, :200]  # Plot first 200 neurons
        plot_pkj_raster(spikes, dt, log_stride, "pkj_raster.png", outdir)
        
        # PKJ mean firing rate over time - shows overall network activity
        print("Computing PKJ mean firing rate...")
        all_pkj_spikes = np_data["PKJ_spikes"]  # Use all PKJ neurons for mean rate
        plot_pkj_mean_rate(all_pkj_spikes, dt, log_stride, "pkj_mean_rate.png", outdir)

    # PF spikes - Parallel Fibers (cortical input)
    if "PF_spikes" in np_data:
        print("Processing PF spikes...")
        N = np_data["PF_spikes"].shape[1]  # Total number of PF neurons
        # Randomly select a subset for visualization (PF population is very large)
        sel = np.random.choice(N, size=min(200, N), replace=False)
        spikes = np_data["PF_spikes"][:, sel]
        plot_pf_raster(spikes, dt, log_stride, "pf_raster.png", outdir)

    # DCN spikes - Deep Cerebellar Nuclei (final output)
    if "DCN_spikes" in np_data:
        print("Processing DCN spikes...")
        spikes = np_data["DCN_spikes"]  # Plot all DCN neurons (there are only 8)
        plot_dcn_raster_with_freq(spikes, dt, log_stride, "dcn_raster.png", outdir)

    # PF→PKJ weights - Synaptic weight evolution (shows learning)
    if "weights" in np_data:
        print("Processing weights...")
        weights = np_data["weights"]  # Weight snapshots over time
        plot_weights(weights, dt, every_steps, outdir)

    # IO voltage trace - Shows gap junction coupling effects
    print("Processing IO voltage trace...")
    plot_io_pair(npz_path, outdir)

    # Clean up: explicitly close the data file
    np_data.close()
    
    # Print timing summary
    elapsed = time.time() - start
    analysis_elapsed = time.time() - analysis_start
    print(f"\n=== Analysis Complete ===")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Processing time: {analysis_elapsed:.1f} seconds")
    print(f"Plots saved to: {outdir}")
    print("=============================")


# --------------------------------------------
# Script entry point
# --------------------------------------------
if __name__ == "__main__":
    """
    When this script is run directly (not imported), it automatically
    analyzes the default simulation output file.
    """
    analyze("cbm_py_output.npz")