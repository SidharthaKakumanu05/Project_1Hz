# Project_1Hz: Cerebellar Microcircuit Simulation

This project simulates a simplified cerebellar microcircuit using GPU-accelerated 
Leaky Integrate-and-Fire (LIF) neurons, plastic synapses, and realistic connectivity. 
The target behavior is that Inferior Olive (IO) neurons fire around 1 Hz as an 
emergent property of the network dynamics.

## 🧠 What This Simulates

The cerebellum is a brain region involved in motor learning and coordination. This 
simulation models the key components:

- **Inferior Olive (IO)**: Provides teaching signals (~1 Hz firing rate)
- **Purkinje Cells (PKJ)**: Main computational units that learn from experience
- **Basket Cells (BC)**: Inhibitory interneurons that regulate PKJ activity
- **Deep Cerebellar Nuclei (DCN)**: Final output neurons
- **Parallel Fibers (PF)**: Excitatory inputs from cortex
- **Mossy Fibers (MF)**: Excitatory inputs from brainstem
- **Climbing Fibers (CF)**: Teaching signals from IO

## 🚦 Control Flow Overview

Below is the **big picture pipeline** of how the code runs:

### 1. **Entry Point**
   - `main.py`
     - Checks for CUDA GPU availability
     - Calls `simulate.run()` to execute the simulation
     - After simulation completes, saves outputs to `cbm_py_output.npz`

### 2. **Simulation Loop**
   - `simulate.py`
     - Loads parameters from `config.py`
     - Builds network wiring with `connectivity.py`
     - Creates neuron populations using `neurons.py`
     - Creates synapse objects using `synapses.py`
     - Initializes inputs (PFs, MFs) with `inputs.py`
     - Applies gap junctions with `coupling.py`
     - Runs the timestep loop:
       - Updates IO, PF, MF, BC, PKJ, DCN in sequence
       - Applies synaptic currents and delays
       - Applies PF→PKJ plasticity rules from `plasticity.py`
       - Records spikes and weights using `recorder.py`
     - Finishes by writing results to `.npz`

### 3. **Analysis**
   - `analysis.py`
     - Loads the `.npz` file produced by simulation
     - Generates publication-ready plots:
       - IO raster with firing rates
       - PKJ raster (first 200 neurons)
       - PF raster (random subset)
       - DCN raster with firing rates
       - PKJ mean firing rate over time
       - PF→PKJ weight traces (individual + mean ± std)
       - Weight distribution analysis
       - IO pair voltage traces (if available)

## 📂 File-by-File Summary

- **`main.py`** → Entry point script, handles GPU check and simulation execution
- **`simulate.py`** → Core simulation engine and main control loop
- **`config.py`** → All simulation parameters (population sizes, neuron configs, synapses, plasticity)
- **`connectivity.py`** → Builds wiring diagrams between neuron populations
- **`neurons.py`** → Defines `NeuronState` class and LIF update function (`lif_step`)
- **`synapses.py`** → Defines `SynapseProj` class for handling delays, weights, conductances
- **`inputs.py`** → Generates PF coinflip spikes and MF Poisson spikes
- **`coupling.py`** → Computes IO gap-junction currents
- **`plasticity.py`** → Applies PF→PKJ LTP/LTD learning rules
- **`recorder.py`** → Records spikes, weights, and timing; saves results to `.npz`
- **`analysis.py`** → Loads `.npz` and produces publication-ready plots
- **`test_analysis_performance.py`** → Performance testing script for analysis

## ▶️ How to Run

### Prerequisites
- NVIDIA GPU with CUDA support
- Python 3.8+ with CuPy, NumPy, Matplotlib
- Optional: Numba for faster analysis (automatically used if available)

### Running the Simulation
1. Run the simulation:
   ```bash
   python3 main.py
   ```
   This will create `cbm_py_output.npz` with all simulation data.

2. Analyze the results:
   ```bash
   python3 analysis.py
   ```
   This will create `analysis_outputs/` directory with publication-ready plots.

### Configuration
All parameters are centralized in `config.py`. Key settings include:
- `T_sec`: Simulation duration (default: 1000 seconds)
- `dt`: Time step (default: 0.1 ms)
- Population sizes (IO, PKJ, BC, DCN, PF, MF)
- Neuron parameters (capacitance, conductance, thresholds)
- Synaptic properties (delays, decay constants, weights)
- Plasticity rules (LTP/LTD windows and scales)

<img width="2579" height="1771" alt="SIM_DIAGRAM" src="https://github.com/user-attachments/assets/8069243c-2711-409e-a006-13160d92dfe2" />
