#!/usr/bin/env python3
"""
Main simulation engine for the cerebellar microcircuit simulation.

This file contains the core simulation loop that orchestrates the entire cerebellar
network simulation. It coordinates neuron updates, synaptic transmission, plasticity,
and data recording to simulate the behavior of a simplified cerebellar microcircuit.

For a freshman undergrad: This is the "brain" of the simulation - it's where everything
comes together and the actual simulation happens!
"""

import os
import time
import cupy as cp
from tqdm import tqdm
from math import log

# Import all the components we need for the simulation
from config import get_config
from neurons import NeuronState, lif_step
from synapses import SynapseProj
from connectivity import build_connectivity
from inputs import init_pf_state, step_pf_coinflip, draw_pf_conductances, step_mf_poisson
from coupling import apply_ohmic_coupling
from plasticity import update_pfpkj_plasticity
from recorder import Recorder


def simulate_current_from_proj(proj, V_post, I_buffer=None):
    """
    Convert a synapse projection's pending conductances into currents on postsynaptic cells.
    
    This function takes the synaptic conductances that are ready to be delivered
    (from the delay buffer) and converts them into actual currents that affect
    the postsynaptic neurons. It uses Ohm's law: I = g * (E_rev - V_post).
    
    Steps:
    1) proj.currents_to_post() gives:
       - g: conductance per arriving event (already aggregated for this step)
       - post_idx: which postsynaptic neuron each conductance goes to
    2) Use conductance-based current: I = g * (E_rev - V_post)
    3) Scatter-add those currents to the postsynaptic neurons they target.

    Parameters
    ----------
    proj : SynapseProj
        The synaptic projection (e.g., PF→PKJ, BC→PKJ)
    V_post : array
        Membrane voltages of postsynaptic neurons
    I_buffer : array, optional
        Pre-allocated buffer for currents (for efficiency)

    Returns
    -------
    I : array
        Current to be applied to each postsynaptic neuron
    """
    g, post_idx = proj.currents_to_post()
    if g.size == 0:  # No events this step
        if I_buffer is not None:
            I_buffer.fill(0.0)
            return I_buffer
        return cp.zeros_like(V_post, dtype=cp.float32)
    
    I_con = g * (proj.E_rev - V_post[post_idx])  # Ohm's law in conductance form
    if I_buffer is not None:
        I_buffer.fill(0.0)
        cp.add.at(I_buffer, post_idx, I_con)     # scatter-accumulate by postsyn neuron index
        return I_buffer
    else:
        I = cp.zeros_like(V_post, dtype=cp.float32)
        cp.add.at(I, post_idx, I_con)            # scatter-accumulate by postsyn neuron index
        return I


def run():
    """
    Main simulation function that runs the complete cerebellar microcircuit simulation.
    
    This function orchestrates the entire simulation process:
    1. Load configuration and set up network structure
    2. Initialize all neuron populations and synaptic connections
    3. Set up input generators and recording systems
    4. Run the main simulation loop
    5. Save results and return simulation information
    
    Returns
    -------
    dict
        Dictionary containing simulation results and output file path
    """
    # --- Load config and basic sizes ---
    # Get all simulation parameters from the centralized configuration
    cfg = get_config()
    dt = cfg["dt"]                    # Time step size (seconds)
    T_steps = cfg["T_steps"]          # Total number of simulation steps
    N_BC, N_PKJ, N_CF, N_DCN = cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"]  # Population sizes

    # --- Build connectivity graph (indices, weights, delays, etc.) ---
    # This creates the "wiring diagram" of which neurons connect to which other neurons
    graph = build_connectivity(cfg)

    # --- Quick IO rheobase / predicted-rate sanity check (first few IOs) ---
    # This calculates the theoretical firing rate of IO neurons to verify our parameters
    # are reasonable before running the full simulation
    print("=== IO Neuron Parameter Check ===")
    for i in range(min(3, N_CF)):
        C = cfg["lif_IO"]["C"]        # Membrane capacitance
        gL = cfg["lif_IO"]["gL"]      # Leak conductance
        Vth = cfg["lif_IO"]["Vth"]    # Spike threshold
        EL = cfg["lif_IO"]["EL"]      # Resting potential
        Vreset = cfg["lif_IO"]["Vreset"]  # Reset potential
        I_bias = cfg["io_bias_current"]    # Bias current

        tau = C / gL                  # Membrane time constant
        I_rheo = gL * (Vth - EL)      # Rheobase current (minimum to cause spiking)
        # T is inter-spike interval for a biased LIF under DC input (closed-form solution)
        T = tau * log((I_bias - gL * (Vreset - EL)) / (I_bias - I_rheo))
        print(f"IO[{i}] tau={tau*1e3:.1f} ms, I_rheo={I_rheo*1e12:.1f} pA, "
              f"I_bias={I_bias*1e12:.1f} pA, predicted rate={1/T:.2f} Hz")
    print("=================================\n")

    # --- Create neuron populations (state vectors: V, refractory, spike flags, etc.) ---
    # Each population is a NeuronState object that holds the current state of all neurons
    # in that population (voltages, refractory periods, spike flags)
    IO  = NeuronState(N_CF,  cfg["lif_IO"])   # Inferior Olive neurons (teaching signals)
    PKJ = NeuronState(N_PKJ, cfg["lif_PKJ"])  # Purkinje cells (main computational units)
    BC  = NeuronState(N_BC,  cfg["lif_BC"])   # Basket cells (inhibitory interneurons)
    DCN = NeuronState(N_DCN, cfg["lif_DCN"])  # Deep Cerebellar Nuclei (output neurons)

    # --- Synapse projections (index lists + dynamics params) ---
    # Each SynapseProj object manages one type of connection between populations
    # It handles synaptic delays, weights, and conductance dynamics
    
    # Climbing fiber -> Purkinje (teaching signals)
    cfpkj = SynapseProj(
        graph["CF_to_PKJ"]["pre_idx"], graph["CF_to_PKJ"]["post_idx"],
        w_init=cp.full(N_CF, 8e-9, dtype=cp.float32),            # CF initial conductance (strong)
        E_rev=cfg["syn_CF_PKJ"]["E_rev"],                        # Excitatory (E_rev = 0)
        tau=cfg["syn_CF_PKJ"]["tau"],                            # Fast decay
        delay_steps=cfg["syn_CF_PKJ"]["delay_steps"]             # Synaptic delay
    )

    # Parallel fiber -> Purkinje (plastic synapse - this is where learning happens!)
    pfpkj = SynapseProj(
        graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["PF_to_PKJ"]["pre_idx"].size, cfg["w_pfpkj_init"], dtype=cp.float32),
        E_rev=cfg["syn_PF_PKJ"]["E_rev"],                        # Excitatory (E_rev = 0)
        tau=cfg["syn_PF_PKJ"]["tau"],                            # Medium decay
        delay_steps=cfg["syn_PF_PKJ"]["delay_steps"]             # Synaptic delay
    )
    # Pre-drawn PF event scaling (per-connection conductance samples)
    pf_con_g = draw_pf_conductances(pfpkj.M, cfg["pf_g_mean"], cfg["pf_g_std"])

    # Parallel fiber -> Basket cell
    pfbc = SynapseProj(
        graph["PF_to_BC"]["pre_idx"], graph["PF_to_BC"]["post_idx"],
        w_init=draw_pf_conductances(graph["PF_to_BC"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"]),
        E_rev=cfg["syn_PF_BC"]["E_rev"],
        tau=cfg["syn_PF_BC"]["tau"],
        delay_steps=cfg["syn_PF_BC"]["delay_steps"]
    )

    # Basket cell -> Purkinje (inhibitory)
    bcpkj = SynapseProj(
        graph["BC_to_PKJ"]["pre_idx"], graph["BC_to_PKJ"]["post_idx"],
        w_init=graph["BC_to_PKJ"]["g"],                     # provided GABA conductances per edge
        E_rev=cfg["syn_BC_PKJ"]["E_rev"],
        tau=cfg["syn_BC_PKJ"]["tau"],
        delay_steps=cfg["syn_BC_PKJ"]["delay_steps"]
    )

    # Purkinje -> DCN (inhibitory)
    pkjdcn = SynapseProj(
        graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
        w_init=cp.full(graph["PKJ_to_DCN"]["pre_idx"].size, 3e-9, dtype=cp.float32),  # reduced inhibition
        E_rev=cfg["syn_PKJ_DCN"]["E_rev"],
        tau=cfg["syn_PKJ_DCN"]["tau"],
        delay_steps=cfg["syn_PKJ_DCN"]["delay_steps"]
    )

    # Mossy fiber -> DCN (excitatory)
    mfdcn = SynapseProj(
        graph["MF_to_DCN"]["pre_idx"], graph["MF_to_DCN"]["post_idx"],
        w_init=cp.full(graph["MF_to_DCN"]["pre_idx"].size, cfg["mf_g_mean"] * 2.0, dtype=cp.float32),  # increased excitation
        E_rev=cfg["syn_MF_DCN"]["E_rev"],
        tau=cfg["syn_MF_DCN"]["tau"],
        delay_steps=cfg["syn_MF_DCN"]["delay_steps"]
    )

    # DCN -> IO (inhibitory feedback)
    dcnio = SynapseProj(
        graph["DCN_to_IO"]["pre_idx"], graph["DCN_to_IO"]["post_idx"],
        w_init=cp.full(graph["DCN_to_IO"]["pre_idx"].size, 1.5e-9, dtype=cp.float32),  # reduced inhibitory strength
        E_rev=cfg["syn_DCN_IO"]["E_rev"],
        tau=cfg["syn_DCN_IO"]["tau"],
        delay_steps=cfg["syn_DCN_IO"]["delay_steps"]
    )

    # --- Precompute decay factors for all synapse exponentials (alpha = exp(-dt/tau)) ---
    for proj, syn_cfg in [
        (cfpkj, cfg["syn_CF_PKJ"]),
        (pfpkj, cfg["syn_PF_PKJ"]),
        (pfbc,  cfg["syn_PF_BC"]),
        (bcpkj, cfg["syn_BC_PKJ"]),
        (pkjdcn,cfg["syn_PKJ_DCN"]),
        (mfdcn, cfg["syn_MF_DCN"]),
        (dcnio, cfg["syn_DCN_IO"]),
    ]:
        proj.set_alpha(cp.exp(-dt / syn_cfg["tau"]).astype(cp.float32))  # cast ensures float32 on GPU

    # --- PF pool (stochastic source) internal state ---
    pf_state = init_pf_state(cfg["N_PF_POOL"], cfg["pf_refrac_steps"])

    # --- IO gap-junction pairs: ring topology (i -> i+1) ---
    if N_CF > 1:
        # pairs[k] = [i, j] means a coupling between IO i and IO j
        pairs = cp.stack([cp.arange(N_CF), cp.roll(cp.arange(N_CF), -1)], axis=1).astype(cp.int32)
    else:
        pairs = cp.zeros((0, 2), dtype=cp.int32)  # no pairs when only one IO

    # --- Recorder setup (stores spikes/weights; logs every `log_stride` steps) ---
    pop_sizes = {
        "PF": cfg["N_PF_POOL"],
        "PKJ": N_PKJ,
        "IO": N_CF,
        "BC": N_BC,
        "DCN": N_DCN,
    }
    rec = Recorder(T_steps, pop_sizes, log_stride=50, rec_weight_every=cfg["rec_weight_every_steps"])
    rec.start_timer()

    # --- Last-spike time trackers (initialize to -inf so first deltas are large/ignored) ---
    last_pf_spike  = cp.full(cfg["N_PF_POOL"], -cp.inf, dtype=cp.float32)
    last_pkj_spike = cp.full(cfg["N_PKJ"],     -cp.inf, dtype=cp.float32)
    last_cf_spike  = cp.full(cfg["N_CF"],      -cp.inf, dtype=cp.float32)

    # Mask of PKJ targets reached by CFs on this step (for LTD gating)
    cf_to_pkj_mask = cp.zeros(N_PKJ, dtype=cp.bool_)
    
    # --- Pre-allocated current buffers to avoid repeated allocations ---
    I_io_buffer = cp.zeros(N_CF, dtype=cp.float32)
    I_bc_buffer = cp.zeros(N_BC, dtype=cp.float32)
    
    # --- Pre-allocated bias current arrays ---
    I_extio = cp.full(N_CF, cfg["io_bias_current"], dtype=cp.float32)
    I_ext_dcn = cp.full(N_DCN, cfg["dcn_bias_current"], dtype=cp.float32)
    I_ext_bc = cp.zeros(N_BC, dtype=cp.float32)
    I_ext_pkj = cp.zeros(N_PKJ, dtype=cp.float32)

    t0 = time.perf_counter()

    # =========================
    # Main simulation loop
    # =========================
    for step in tqdm(range(T_steps), desc="Simulating", unit="steps"):
        sim_t = step * dt  # current sim time in seconds

        # ----- Inferior Olive (IO) update -----
        dcnio.step_decay()                                 # decay DCN->IO conductances
        I_gap   = apply_ohmic_coupling(IO.V, pairs, cfg["g_gap_IO"])          # gap-junction currents
        I_dcnio = simulate_current_from_proj(dcnio, IO.V, I_io_buffer)  # DCN->IO inhibition
        IO = lif_step(IO, I_syn=I_gap + I_dcnio, I_ext=I_extio, dt=dt, lif_cfg=cfg["lif_IO"])
        cf_spike = IO.spike  # CF spikes are IO spikes (no copy needed)
        if cp.any(cf_spike):
            cfpkj.enqueue_from_pre_spikes(cf_spike)  # schedule CF→PKJ events with synaptic delay

        # ----- Parallel fibers (PF) & Mossy fibers (MF) -----
        # PF are coin-flip (Bernoulli) per-step with refractory logic inside pf_state
        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate_hz"], dt)
        rec.log_spikes(step, "PF", pf_spike_pool)
        if cp.any(pf_spike_pool):
            # PF→PKJ uses per-connection conductance samples (pf_con_g) as scaling on enqueued events
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            # PF→BC uses its own conductance draws baked into pfbc.w
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)

        # Mossy fibers are independent Poisson to DCN
        mf_spike = step_mf_poisson(cfg["N_DCN"], cfg["mf_rate_hz"], dt)
        if cp.any(mf_spike):
            mfdcn.enqueue_from_pre_spikes(mf_spike)

        # ----- Basket cells (BC) -----
        pfbc.step_decay()                                  # exponential decay of PF→BC conductances
        I_bc = simulate_current_from_proj(pfbc, BC.V, I_bc_buffer)      # turn scheduled events into currents on BC
        BC = lif_step(BC, I_syn=I_bc, I_ext=I_ext_bc, dt=dt, lif_cfg=cfg["lif_BC"])
        if cp.any(BC.spike):
            bcpkj.enqueue_from_pre_spikes(BC.spike)        # BC inhibition scheduled onto PKJ

        # ----- Purkinje cells (PKJ) -----
        # Decay all incoming synaptic projections before gathering their currents
        cfpkj.step_decay()
        bcpkj.step_decay()
        pfpkj.step_decay()
        I_pkj = (
            simulate_current_from_proj(cfpkj, PKJ.V) +
            simulate_current_from_proj(bcpkj, PKJ.V) +
            simulate_current_from_proj(pfpkj, PKJ.V)
        )
        PKJ = lif_step(PKJ, I_syn=I_pkj, I_ext=I_ext_pkj, dt=dt, lif_cfg=cfg["lif_PKJ"])

        # ----- Deep cerebellar nuclei (DCN) -----
        if cp.any(PKJ.spike):
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)      # PKJ inhibit DCN when they spike
        pkjdcn.step_decay()
        mfdcn.step_decay()
        I_dcn = simulate_current_from_proj(pkjdcn, DCN.V) + simulate_current_from_proj(mfdcn, DCN.V)
        DCN = lif_step(DCN, I_syn=I_dcn, I_ext=I_ext_dcn, dt=dt, lif_cfg=cfg["lif_DCN"])
        
        # DCN -> IO inhibition (feedback loop)
        if cp.any(DCN.spike):
            dcnio.enqueue_from_pre_spikes(DCN.spike)       # DCN inhibit IO when they spike

        # ----- CF→PKJ gating mask for plasticity (which PKJ received a CF this step?) -----
        cf_to_pkj_mask.fill(False)
        if cp.any(cf_spike):
            # cfpkj.post_idx is aligned with CF pres; boolean-index with cf_spike to get PKJ targets
            pkj_targets = cfpkj.post_idx[cf_spike]    # tricky bit: boolean mask selects posts of spiking CFs
            cf_to_pkj_mask[pkj_targets] = True

        # ----- PF→PKJ plasticity (uses last spike times + CF gating mask) -----
        # Only run plasticity every few steps to improve performance
        if step % cfg["plasticity_every_steps"] == 0:
            pfpkj.w, last_pf_spike, last_pkj_spike, last_cf_spike = update_pfpkj_plasticity(
                pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
                pf_spike_pool, PKJ.spike, cf_to_pkj_mask,
                sim_t, cfg,
                last_pf_spike, last_pkj_spike, last_cf_spike
            )

        # ----- Record spikes and (optionally) weights -----
        rec.log_spikes(step, "IO", IO.spike)
        rec.log_spikes(step, "PKJ", PKJ.spike)
        rec.log_spikes(step, "BC", BC.spike)
        rec.log_spikes(step, "DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)

    # --- Timing summary and save ---
    elapsed, steps_per_sec = rec.stop_and_summary(T_steps)

    out_npz = "cbm_py_output.npz"   # final output filename
    rec.finalize_npz(out_npz)       # pack arrays and metadata

    return {
        "out_npz": os.path.abspath(out_npz),
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
    }


if __name__ == "__main__":
    # Allow running this file directly for a quick smoke test
    info = run()
    print("Saved outputs to", info["out_npz"])