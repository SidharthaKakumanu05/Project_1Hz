# Project_1Hz

This project simulates a simplified cerebellar microcircuit using
GPU-accelerated leaky integrate-and-fire (LIF) neurons, plastic synapses,
and realistic connectivity. The target behavior is that Inferior Olive (IO)
neurons fire around 1 Hz as an emergent property.

---

## 🚦 Control Flow Overview

Below is the **big picture pipeline** of how the code runs:

1. **Entry point**
   - `main.py`
     - Calls `simulate.run()` to execute the simulation.
     - After simulation completes, saves outputs to `cbm_py_output.npz`.

2. **Simulation loop**
   - `simulate.py`
     - Loads parameters from `config.py`.
     - Builds network wiring with `connectivity.py`.
     - Creates neuron populations using `neurons.py`.
     - Creates synapse objects using `synapses.py`.
     - Initializes inputs (PFs, MFs) with `inputs.py`.
     - Applies gap junctions with `coupling.py`.
     - Runs the timestep loop:
       - Updates IO, PF, MF, BC, PKJ, DCN in sequence.
       - Applies synaptic currents and delays.
       - Applies PF→PKJ plasticity rules from `plasticity.py`.
       - Records spikes and weights using `recorder.py`.
     - Finishes by writing results to `.npz`.

3. **Analysis**
   - `analysis.py`
     - Loads the `.npz` file produced by simulation.
     - Generates plots:
       - IO raster with firing rates
       - PKJ raster
       - PF raster (subset)
       - PF→PKJ weight traces (individual + mean ± std)
       - IO pair voltage traces (if available)

---

## 📂 File-by-File Summary

- **`main.py`** → simple entry script, runs simulation.  
- **`simulate.py`** → core simulation engine and control loop.  
- **`config.py`** → all parameters (pop sizes, neuron configs, synapses, plasticity).  
- **`connectivity.py`** → builds wiring diagrams between populations.  
- **`neurons.py`** → defines `NeuronState` and the LIF update (`lif_step`).  
- **`synapses.py`** → defines `SynapseProj` for handling delays, weights, conductances.  
- **`inputs.py`** → generates PF coinflip spikes and MF Poisson spikes.  
- **`coupling.py`** → computes IO gap-junction currents.  
- **`plasticity.py`** → applies PF→PKJ LTP/LTD rules.  
- **`recorder.py`** → records spikes, weights, and timing; saves results to `.npz`.  
- **`analysis.py`** → loads `.npz` and produces plots for inspection.  

---

## ▶️ How to Run

1. Run the simulation:
   ```bash
   python main.py