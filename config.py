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
    cfg["N_CF"]  = 16      # climbing fibers - provide natural teaching signals from IO for LTD/LTP
    cfg["N_DCN"] = 8       # deep cerebellar nuclei - final output of the cerebellum
    cfg["N_PF_POOL"] = 4096  # parallel fiber pool size - massive excitatory input to Purkinje cells

    # -----------------------------
    # Time settings
    # -----------------------------
    # These control the temporal aspects of the simulation.
    # Smaller dt = more accurate but slower simulation.
    
    cfg["dt"] = 1e-3                # time step (0.2 ms) - CbmSim likely used this based on scaling
    cfg["T_sec"] = 60               # total simulation length in seconds - how long to run
    cfg["T_steps"] = int(cfg["T_sec"] / cfg["dt"])  # number of simulation steps - calculated automatically
    cfg["seed"] = 12345               # random seed for reproducibility - same seed = same results

    # -----------------------------
    # Leaky Integrate-and-Fire (LIF) neuron parameters
    # Units: capacitance (F), conductance (S), voltage (V)
    # -----------------------------
    # Each neuron type has different electrical properties that determine how it behaves.
    # Think of these as the "personality" of each neuron type!
    
    # Inferior Olive (IO) neurons - INCREASED leak for spontaneous firing without noise
    cfg["lif_IO"] = dict(gL=0.25, EL=-60.0,                     # INCREASED from 0.15 to 0.25 for spontaneous firing
                         Vth=-57.4, Vreset=-70.0,                # CbmSim: threshRestIO=-57.4
                         refrac_steps=int(0.002 / cfg["dt"]),    # 2 ms refractory period
                         thresh_decay=0.00499, thresh_rest=-57.4, # CbmSim: threshDecIO=0.00499, threshRestIO=-57.4
                         thresh_max=10.0, eNCtoIO=-80.0)         # CbmSim: threshMaxIO=10.0, eNCtoIO=-80.0

    # Purkinje cells - CbmSim exact parameters
    cfg["lif_PKJ"] = dict(gL=0.2, EL=-60.0,                     # Raw CbmSim value: rawGLeakPC = 0.2
                          Vth=-60.62, Vreset=-72.0,              # CbmSim: threshRestPC=-60.62
                          refrac_steps=int(0.002 / cfg["dt"]),   # 2 ms refractory period
                          thresh_decay=0.1535, thresh_rest=-60.62, # CbmSim: threshDecPC=0.1535, threshRestPC=-60.62
                          thresh_max=-48.0, eBCtoPC=-80.0, eSCtoPC=-80.0) # CbmSim: threshMaxPC=-48.0, eBCtoPC=-80.0, eSCtoPC=-80.0

    # Basket cells - CbmSim exact parameters
    cfg["lif_BC"]  = dict(gL=0.13, EL=-70.0,                   # CbmSim: gLeakBC = rawGLeakBC = 0.13, eLeakBC=-70.0
                          Vth=-65.0, Vreset=-70.0,              # CbmSim: threshRestBC=-65.0
                          refrac_steps=int(0.0015 / cfg["dt"]), # 1.5 ms refractory period
                          thresh_decay=0.0952, thresh_rest=-65.0, # CbmSim: threshDecBC=0.0952, threshRestBC=-65.0
                          thresh_max=0.0, ePCtoBC=-70.0)        # CbmSim: threshMaxBC=0.0, ePCtoBC=-70.0

    # Deep Cerebellar Nuclei - EXTREMELY HARD TO EXCITE (pushing for IO 1Hz)
    cfg["lif_DCN"] = dict(gL=1.0, EL=-65.0,                    # EXTREME from 0.5 to 1.0 (extremely hard to excite)
                          Vth=-55.0, Vreset=-62.0,              # EXTREME threshold from -60.0 to -55.0 (extremely hard to reach)
                          refrac_steps=int(0.01 / cfg["dt"]),   # VERY LONG refractory period from 5ms to 10ms
                          thresh_decay=0.1813, thresh_rest=-55.0, # Updated thresh_rest to match new threshold
                          thresh_max=-40.0, ePCtoNC=-80.0)      # CbmSim: threshMaxNC=-40.0, ePCtoNC=-80.0

    # -----------------------------
    # Synapse model parameters
    # tau = decay time constant (s) - how long synaptic effects last
    # E_rev = reversal potential (V) - determines if synapse is excitatory or inhibitory
    # delay_steps = synaptic delay (# of dt steps) - how long it takes signal to travel
    # -----------------------------
    # Synapses are the connections between neurons. Each type of connection has
    # different properties that determine how signals are transmitted.
    
    # Parallel Fiber → Purkinje Cell (excitatory, fast) - CbmSim exact values
    cfg["syn_PF_PKJ"] = dict(tau=4.15e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))  # CbmSim: gDecTauGRtoPC=4.15
    
    # Parallel Fiber → Basket Cell (excitatory, fast)
    cfg["syn_PF_BC"]  = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))  # Same as PF→PKJ
    
    # Basket Cell → Purkinje Cell (inhibitory, slower) - CbmSim exact values
    cfg["syn_BC_PKJ"] = dict(tau=7e-3, E_rev=-80.0, delay_steps=int(1e-3 / cfg["dt"]))  # CbmSim: gDecTauBCtoPC=7.0
    
    # Climbing Fiber → Purkinje Cell (excitatory, very fast) - Natural teaching signals
    cfg["syn_CF_PKJ"] = dict(tau=2e-3,  E_rev=0.0,    delay_steps=int(2e-3 / cfg["dt"]))  # Fast decay, longer delay
    
    # Purkinje Cell → Deep Cerebellar Nuclei (inhibitory, slow) - CbmSim exact values
    cfg["syn_PKJ_DCN"]= dict(tau=4.15e-3, E_rev=-80.0, delay_steps=int(2e-3 / cfg["dt"]))  # CbmSim: gDecTauPCtoNC=4.15
    
    # Mossy Fiber → Deep Cerebellar Nuclei (excitatory, medium)
    cfg["syn_MF_DCN"] = dict(tau=6e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))  # Medium decay
    
    # Deep Cerebellar Nuclei → Inferior Olive (inhibitory, very slow) - CbmSim exact values
    cfg["syn_DCN_IO"] = dict(tau=300e-3, E_rev=-80.0, delay_steps=int(3e-3 / cfg["dt"]))  # CbmSim: gIncTauNCtoIO=300.0

    # -----------------------------
    # Input rates and conductance distributions
    # -----------------------------
    # These control how the external inputs (PF and MF) behave.
    # The cerebellum receives constant input from the cortex and brainstem.
    
    # Parallel Fiber inputs - EXACT CbmSim values for proper PKJ firing
    cfg["pf_rate_hz"] = 30.0                        # EXACT CbmSim value: 30 Hz
    cfg["pf_refrac_steps"] = int(0.005 / cfg["dt"]) # PF refractory period (5 ms) - prevents unrealistic high firing
    cfg["pf_g_mean"] = 0.0000055                    # EXACT CbmSim value: gIncGRtoPC=0.55e-05
    cfg["pf_g_std"]  = 0.00000275                   # EXACT CbmSim value: half of mean

    # Mossy Fiber inputs - REDUCED to make DCN harder to excite
    cfg["mf_rate_hz"] = 10.0                        # REDUCED from 25.0 to 10.0 Hz
    cfg["mf_g_mean"] = 1.0                          # REDUCED from 2.35 to 1.0
    cfg["mf_g_std"]  = 0.1                          # REDUCED from 0.25 to 0.1

    # Basket Cell connection strengths - EXACT CbmSim values for proper PKJ control
    cfg["bc_g_mean"] = 0.0003                       # BC conductance mean (S) - CbmSim: gIncBCtoPC=0.0003
    cfg["bc_g_std"]  = 0.00015                      # BC conductance std dev - half of mean
    cfg["bc_pkj_g_scale"] = 1.0                     # EXACT CbmSim scaling - no reduction

    # -----------------------------
    # PF→PKJ initial weights & limits
    # -----------------------------
    # These control the learning that happens at PF→PKJ synapses.
    # The cerebellum learns by changing these connection strengths!
    
    cfg["w_pfpkj_init"] = 1.0   # initial weight - should start at 1.0
    cfg["w_min"] = 0.2          # lower bound - synapses can't get weaker than this
    cfg["w_max"] = 3.0          # upper bound - synapses can't get stronger than this
    # No weight leak - CbmSim uses LTP/LTD equilibrium for forgetting
    # cfg["w_leak"] = 0.0        # CbmSim doesn't use weight leak

    # -----------------------------
    # IO (Inferior Olive) excitability - CbmSim style
    # -----------------------------
    # IO neurons fire spontaneously at 2-4 Hz due to their leak conductance properties.
    # With DCN inhibition, they will fire at ~1 Hz as required.
    
    cfg["io_noise_std"] = 0.0             # REMOVED noise - IO should fire spontaneously from leak + gap junctions
    cfg["io_bias_jitter_std"] = 0.0      # REMOVED bias jitter - not used anyway
    # No bias current - IO must fire spontaneously from leak conductances only
    cfg["io_thresh_decay"] = 0.005        # threshold decay rate (CbmSim: threshDecIO)
    cfg["io_thresh_rest"] = -0.0574       # resting threshold (CbmSim: threshRestIO)

    # -----------------------------
    # DCN (Deep Cerebellar Nuclei) excitability
    # -----------------------------
    # DCN neurons are driven purely by synaptic inputs (MF excitation + PC inhibition)
    # No bias current - matches CbmSim exactly
    cfg["dcn_bias_current"] = 0.0         # No bias current - driven by synaptic inputs only

    # -----------------------------
    # Gap-junction coupling between IO neurons
    # -----------------------------
    # IO neurons are electrically coupled to each other, which helps synchronize them.
    # CRITICAL: REDUCED gap junction coupling to reduce IO firing
    # Strong coupling might be causing IO to fire too much
    cfg["g_gap_IO"] = 0.01      # REDUCED from 0.05 to 0.01

    # -----------------------------
    # Plasticity parameters
    # -----------------------------
    # These control how the PF→PKJ synapses learn (LTP and LTD).
    # This is the key to cerebellar learning!
    
    cfg["ltd_window"] = dict(t_pre_cf=0.1)   # LTD window = 100 ms - PF must precede CF by this much
    cfg["ltp_window"] = dict(t_pre_cf=0.9)   # LTP window = 900 ms - PF can precede CF by up to this much
    cfg["ltd_scale"]  = 0.00275              # LTD decrement size - CbmSim: synLTDStepSizeGRtoPC = -0.00275
    cfg["ltp_scale"]  = 0.00030556           # LTP increment size - CbmSim: synLTPStepSizeGRtoPC = 0.00030556

    # -----------------------------
    # Recording frequency
    # -----------------------------
    # How often to save data during the simulation.
    # More frequent recording = more detailed data but larger files.
    
    cfg["rec_weight_every_steps"] = int(0.1 / cfg["dt"])  # record weights every 100 ms
    cfg["plasticity_every_steps"] = int(0.01 / cfg["dt"])  # run plasticity every 10 ms

    return cfg