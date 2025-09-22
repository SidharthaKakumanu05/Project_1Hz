#!/usr/bin/env python3
"""
Configuration file for the cerebellar microcircuit simulation.

This file contains ALL the parameters needed to run the simulation.
Think of it as the "settings" or "recipe" for the entire experiment.

For a freshman undergrad: This is where you would change things like:
- How long to run the simulation
- How many neurons to simulate
- How strong the connections are
- What the learning rules are

IMPORTANT: This is the "single source of truth" - all parameters are defined here!
"""

import cupy as cp

def get_config():
    """
    Return a dictionary of all simulation parameters.
    
    This function is the single source of truth for ALL simulation settings.
    Instead of having parameters scattered throughout the code, everything is
    centralized here. This makes it easy to:
    - Find and modify parameters
    - Ensure consistency across the codebase
    - Document what each parameter does
    - Run different experiments by changing just this file
    
    Returns:
        cfg (dict): Dictionary containing all simulation parameters organized by category
    """
    cfg = {}

    # -----------------------------
    # Network population sizes
    # -----------------------------
    # These numbers determine how many neurons of each type we simulate.
    # The cerebellum has a very specific ratio of cell types - this reflects that!
    
    cfg["N_BC"]  = 512     # basket cells - inhibitory interneurons that inhibit Purkinje cells
    cfg["N_PKJ"] = 128     # Purkinje cells - the main output neurons of the cerebellar cortex
    cfg["N_CF"]  = 16      # climbing fibers - provide teaching signals from the inferior olive
    cfg["N_DCN"] = 8       # deep cerebellar nuclei - final output of the cerebellum
    cfg["N_PF_POOL"] = 4096  # parallel fiber pool size - massive excitatory input to Purkinje cells

    # -----------------------------
    # Time settings
    # -----------------------------
    # These control the temporal aspects of the simulation.
    # Smaller dt = more accurate but slower simulation.
    
    cfg["dt"] = 1e-4                  # time step (0.1 ms) - how fine-grained our simulation is
    cfg["T_sec"] = 1000               # total simulation length in seconds - how long to run
    cfg["T_steps"] = int(cfg["T_sec"] / cfg["dt"])  # number of simulation steps - calculated automatically
    cfg["seed"] = 12345               # random seed for reproducibility - same seed = same results

    # -----------------------------
    # Leaky Integrate-and-Fire (LIF) neuron parameters
    # Units: capacitance (F), conductance (S), voltage (V)
    # -----------------------------
    # Each neuron type has different electrical properties that determine how it behaves.
    # Think of these as the "personality" of each neuron type!
    
    # Inferior Olive (IO) neurons - these are special because they provide teaching signals
    cfg["lif_IO"] = dict(C=1.0e-9, gL=5.4e-9, EL=-0.067,        # C=capacitance, gL=leak conductance, EL=resting potential
                         Vth=-0.052, Vreset=-0.070,              # Vth=spike threshold, Vreset=voltage after spike
                         refrac_steps=int(0.002 / cfg["dt"]))    # 2 ms refractory period (can't spike again immediately)

    # Purkinje cells - the main computational units of the cerebellum
    cfg["lif_PKJ"] = dict(C=250e-12, gL=14e-9, EL=-0.068,       # Different from IO - more excitable
                          Vth=-0.052, Vreset=-0.072,            # Similar threshold but different reset
                          refrac_steps=int(0.002 / cfg["dt"]))  # Same refractory period

    # Basket cells - fast inhibitory interneurons
    cfg["lif_BC"]  = dict(C=180e-12, gL=10e-9, EL=-0.065,       # Small, fast cells
                          Vth=-0.050, Vreset=-0.070,            # Lower threshold = more likely to spike
                          refrac_steps=int(0.0015 / cfg["dt"])) # Shorter refractory = can spike more often

    # Deep Cerebellar Nuclei - final output neurons
    cfg["lif_DCN"] = dict(C=220e-12, gL=10e-9, EL=-0.060,       # Medium-sized output neurons
                          Vth=-0.048, Vreset=-0.062,            # Lowest threshold = most excitable
                          refrac_steps=int(0.002 / cfg["dt"]))  # Standard refractory period

    # -----------------------------
    # Synapse model parameters
    # tau = decay time constant (s) - how long synaptic effects last
    # E_rev = reversal potential (V) - determines if synapse is excitatory or inhibitory
    # delay_steps = synaptic delay (# of dt steps) - how long it takes signal to travel
    # -----------------------------
    # Synapses are the connections between neurons. Each type of connection has
    # different properties that determine how signals are transmitted.
    
    # Parallel Fiber → Purkinje Cell (excitatory, fast)
    cfg["syn_PF_PKJ"] = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))  # E_rev=0 means excitatory
    
    # Parallel Fiber → Basket Cell (excitatory, fast)
    cfg["syn_PF_BC"]  = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))  # Same as PF→PKJ
    
    # Basket Cell → Purkinje Cell (inhibitory, slower)
    cfg["syn_BC_PKJ"] = dict(tau=10e-3, E_rev=-0.075, delay_steps=int(1e-3 / cfg["dt"]))  # E_rev<0 means inhibitory
    
    # Climbing Fiber → Purkinje Cell (excitatory, very fast)
    cfg["syn_CF_PKJ"] = dict(tau=2e-3,  E_rev=0.0,    delay_steps=int(2e-3 / cfg["dt"]))  # Fast decay, longer delay
    
    # Purkinje Cell → Deep Cerebellar Nuclei (inhibitory, slow)
    cfg["syn_PKJ_DCN"]= dict(tau=12e-3, E_rev=-0.075, delay_steps=int(2e-3 / cfg["dt"]))  # Slow, inhibitory
    
    # Mossy Fiber → Deep Cerebellar Nuclei (excitatory, medium)
    cfg["syn_MF_DCN"] = dict(tau=6e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))  # Medium decay
    
    # Deep Cerebellar Nuclei → Inferior Olive (inhibitory, very slow)
    cfg["syn_DCN_IO"] = dict(tau=15e-3, E_rev=-0.075, delay_steps=int(3e-3 / cfg["dt"]))  # Longest decay and delay

    # -----------------------------
    # Input rates and conductance distributions
    # -----------------------------
    # These control how the external inputs (PF and MF) behave.
    # The cerebellum receives constant input from the cortex and brainstem.
    
    # Parallel Fiber inputs - represent cortical input to the cerebellum
    cfg["pf_rate_hz"] = 30.0                        # average PF firing rate (Hz) - realistic cortical firing rate
    cfg["pf_refrac_steps"] = int(0.005 / cfg["dt"]) # PF refractory period (5 ms) - prevents unrealistic high firing
    cfg["pf_g_mean"] = 4e-10                        # mean PF conductance (S) - how strong the connections are
    cfg["pf_g_std"]  = 2e-10                        # std dev of PF conductance - adds realistic variability

    # Mossy Fiber inputs - represent brainstem input to the cerebellum
    cfg["mf_rate_hz"] = 50.0                        # average MF firing rate (Hz) - higher than PF
    cfg["mf_g_mean"] = 4e-9                         # mean MF conductance (S) - 10x stronger than PF
    cfg["mf_g_std"]  = 1e-9                         # std dev of MF conductance - less variability than PF

    # Basket Cell connection strengths - these are inhibitory connections
    cfg["bc_g_mean"] = 5e-9                         # BC conductance mean (S) - strong inhibition
    cfg["bc_g_std"]  = 1e-9                         # BC conductance std dev - some variability
    cfg["bc_pkj_g_scale"] = 0.5                     # scaling for BC→PKJ inhibition - reduces strength by half

    # -----------------------------
    # PF→PKJ initial weights & limits
    # -----------------------------
    # These control the learning that happens at PF→PKJ synapses.
    # The cerebellum learns by changing these connection strengths!
    
    cfg["w_pfpkj_init"] = 1.0   # initial weight - all synapses start at the same strength
    cfg["w_min"] = 0.2          # lower bound - synapses can't get weaker than this
    cfg["w_max"] = 3.0          # upper bound - synapses can't get stronger than this
    cfg["w_leak"] = 1e-6        # weight decay rate - prevents weights from drifting upward forever

    # -----------------------------
    # IO (Inferior Olive) excitability
    # -----------------------------
    # The IO neurons need special treatment because they're the "teachers" in the cerebellum.
    # They need to be excitable enough to fire at ~1 Hz.
    
    cfg["io_bias_current"] = 83e-12       # constant bias current (pA) - keeps IO neurons ready to fire
    cfg["io_noise_std"] = 2e-12           # current noise - adds realistic variability
    cfg["io_bias_jitter_std"] = 0.3e-12   # static jitter per-neuron - each neuron is slightly different

    # -----------------------------
    # DCN (Deep Cerebellar Nuclei) excitability
    # -----------------------------
    # DCN neurons need to be balanced between excitation and inhibition.
    # DCN rheobase: gL * (Vth - EL) = 10e-9 * (-0.048 - (-0.060)) = 1.2e-10 A
    # Add moderate bias current to keep DCN neurons near threshold
    cfg["dcn_bias_current"] = 0.8e-10     # bias current (80 pA, below rheobase) - keeps them responsive

    # -----------------------------
    # Gap-junction coupling between IO neurons
    # -----------------------------
    # IO neurons are electrically coupled to each other, which helps synchronize them.
    cfg["g_gap_IO"] = 1e-10   # coupling strength - not too strong, avoids full synchronization

    # -----------------------------
    # Plasticity parameters
    # -----------------------------
    # These control how the PF→PKJ synapses learn (LTP and LTD).
    # This is the key to cerebellar learning!
    
    cfg["ltd_window"] = dict(t_pre_cf=0.05)  # LTD window = 50 ms - PF must precede CF by this much
    cfg["ltp_window"] = dict(t_pre_cf=0.45)  # LTP window = 450 ms - PF can precede CF by up to this much
    cfg["ltd_scale"]  = 9e-4                 # LTD decrement size - how much weights decrease
    cfg["ltp_scale"]  = 1e-4                 # LTP increment size - how much weights increase

    # -----------------------------
    # Recording frequency
    # -----------------------------
    # How often to save data during the simulation.
    # More frequent recording = more detailed data but larger files.
    
    cfg["rec_weight_every_steps"] = int(0.1 / cfg["dt"])  # record weights every 100 ms
    cfg["plasticity_every_steps"] = int(0.01 / cfg["dt"])  # run plasticity every 10 ms

    return cfg