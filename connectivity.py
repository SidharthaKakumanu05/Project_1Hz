#!/usr/bin/env python3
"""
Network connectivity construction for the cerebellar microcircuit simulation.

This file builds the "wiring diagram" of the cerebellar network - it determines
which neurons connect to which other neurons. The connectivity patterns are
based on known anatomical and physiological properties of the real cerebellum.

For a freshman undergrad: This is like drawing the "circuit diagram" of the
cerebellar network - it shows how all the different neuron types are connected!
"""

import cupy as cp
import numpy as np

def build_connectivity(cfg):
    """
    Build all the connectivity matrices for the cerebellar network.
    
    This function creates the "wiring diagram" that defines how different
    neuron populations connect to each other. Each connection type has
    specific properties based on real cerebellar anatomy:
    
    - PF→PKJ: Massive convergent input (thousands of PFs per PKJ)
    - PF→BC: Convergent input to inhibitory interneurons
    - CF→PKJ: One-to-one teaching signals from inferior olive
    - BC→PKJ: Local inhibitory feedback
    - PKJ→DCN: Output from cerebellar cortex
    - MF→DCN: Direct excitatory input
    - DCN→IO: Feedback inhibition to teaching signal source

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing population sizes and random seed

    Returns
    -------
    graph : dict
        Dictionary where each key corresponds to a projection (e.g., "PF_to_PKJ")
        and contains arrays of presynaptic indices, postsynaptic indices, and weights.
        This defines the complete connectivity of the network.
    """
    N_BC, N_PKJ, N_CF, N_DCN = cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"]
    rng = np.random.default_rng(cfg["seed"])  # reproducible random generator
    graph = {}

    # -----------------------------
    # PF → PKJ (random convergent)
    # -----------------------------
    # This is the most important connection in the cerebellum!
    # Each PKJ receives inputs from thousands of PFs (massive convergence)
    # In real cerebellum: ~200,000 PFs per PKJ; here we use ~1000 for computational efficiency
    pf_conn = cfg["N_PF_POOL"] // 4  # Each PKJ gets input from 1/4 of all PFs
    pf_pre = rng.integers(0, cfg["N_PF_POOL"], size=pf_conn * N_PKJ, dtype=np.int32)
    pf_post = np.repeat(np.arange(N_PKJ, dtype=np.int32), pf_conn)
    graph["PF_to_PKJ"] = {
        "pre_idx": cp.asarray(pf_pre),    # Which PFs connect to each synapse
        "post_idx": cp.asarray(pf_post),  # Which PKJ receives each connection
    }

    # -----------------------------
    # PF → BC (random convergent)
    # -----------------------------
    # Basket cells also receive PF input, but less than PKJs
    # This allows them to be activated by the same cortical input that drives PKJs
    pfbc_conn = cfg["N_PF_POOL"] // 8  # Each BC gets input from 1/8 of all PFs
    pfbc_pre = rng.integers(0, cfg["N_PF_POOL"], size=pfbc_conn * N_BC, dtype=np.int32)
    pfbc_post = np.repeat(np.arange(N_BC, dtype=np.int32), pfbc_conn)
    graph["PF_to_BC"] = {
        "pre_idx": cp.asarray(pfbc_pre),    # Which PFs connect to each BC
        "post_idx": cp.asarray(pfbc_post),  # Which BC receives each connection
    }

    # -----------------------------
    # CF → PKJ (climbing fibers)
    # -----------------------------
    # Climbing fibers provide teaching signals from the inferior olive
    # CRITICAL: Must match CbmSim exactly! CbmSim uses num_p_io_from_io_to_pc = 8
    # Each CF connects to 8 PKJ cells in blocks: CF 0 → PKJ 0-7, CF 1 → PKJ 8-15, etc.
    # This block pattern prevents PKJ drift by ensuring all PKJ cells get CF teaching signals
    
    # Calculate how many PKJ cells each CF should connect to
    pkj_per_cf = N_PKJ // N_CF  # Should be 8 for N_PKJ=128, N_CF=16
    
    cf_pre = []   # CF indices
    cf_post = []  # PKJ indices
    
    for cf_idx in range(N_CF):
        for j in range(pkj_per_cf):
            pkj_idx = cf_idx * pkj_per_cf + j
            if pkj_idx < N_PKJ:  # Make sure we don't exceed PKJ count
                cf_pre.append(cf_idx)
                cf_post.append(pkj_idx)
    
    graph["CF_to_PKJ"] = {
        "pre_idx": cp.asarray(cf_pre, dtype=cp.int32),    # Which CF provides the teaching signal
        "post_idx": cp.asarray(cf_post, dtype=cp.int32),  # Which PKJ receives the teaching signal
    }

    # -----------------------------
    # BC → PKJ (inhibitory, sparse)
    # -----------------------------
    # Basket cells provide local inhibitory feedback to Purkinje cells
    # This creates a "lateral inhibition" effect - active PKJs inhibit their neighbors
    bc_conn = N_BC // 4  # Each PKJ receives inhibition from 1/4 of all BCs
    bc_pre  = rng.integers(0, N_BC, size=bc_conn * N_PKJ, dtype=np.int32)
    bc_post = np.repeat(np.arange(N_PKJ, dtype=np.int32), bc_conn)

    # Synaptic strengths: sampled from normal distribution, scaled, then clipped ≥ 0
    # This gives realistic variability in inhibitory strength
    bc_g = rng.normal(cfg["bc_g_mean"], cfg["bc_g_std"], size=bc_conn * N_PKJ)
    bc_g = bc_g * cfg["bc_pkj_g_scale"]  # Scale down the inhibition strength
    bc_g = cp.asarray(np.clip(bc_g, 0, None), dtype=cp.float32)  # No negative conductances

    graph["BC_to_PKJ"] = {
        "pre_idx": cp.asarray(bc_pre),    # Which BC provides inhibition
        "post_idx": cp.asarray(bc_post),  # Which PKJ receives inhibition
        "g": bc_g,                        # inhibitory conductances (negative effect)
    }

    # -----------------------------
    # PKJ → DCN (random sparse inhibition)
    # -----------------------------
    # Purkinje cells provide the main output from cerebellar cortex to deep nuclei
    # Each DCN neuron receives input from multiple PKJs (convergent inhibition)
    pkj_conn = N_PKJ // 8  # Each DCN gets input from 1/8 of all PKJs
    pkj_pre  = rng.integers(0, N_PKJ, size=pkj_conn * N_DCN, dtype=np.int32)
    pkj_post = np.repeat(np.arange(N_DCN, dtype=np.int32), pkj_conn)
    graph["PKJ_to_DCN"] = {
        "pre_idx": cp.asarray(pkj_pre),    # Which PKJ provides output
        "post_idx": cp.asarray(pkj_post),  # Which DCN receives the output
    }

    # -----------------------------
    # MF → DCN (mossy fibers)
    # -----------------------------
    # Mossy fibers provide direct excitatory input to deep cerebellar nuclei
    # For simplicity, we use a 1:1 mapping (one MF per DCN neuron)
    # In reality, there would be more MFs than DCN neurons
    graph["MF_to_DCN"] = {
        "pre_idx": cp.arange(N_DCN, dtype=cp.int32),  # MF indices: 0, 1, 2, ...
        "post_idx": cp.arange(N_DCN, dtype=cp.int32), # DCN indices: 0, 1, 2, ...
    }

    # -----------------------------
    # DCN → IO (inhibitory feedback)
    # -----------------------------
    # Deep cerebellar nuclei provide inhibitory feedback to inferior olive
    # This creates a negative feedback loop that helps stabilize the system
    # Each IO neuron receives input from multiple DCN neurons
    dcn_conn = max(1, N_DCN // 2)  # ensure at least 1 connection per IO
    dcn_pre  = rng.integers(0, N_DCN, size=dcn_conn * N_CF, dtype=np.int32)
    dcn_post = np.repeat(np.arange(N_CF, dtype=np.int32), dcn_conn)
    graph["DCN_to_IO"] = {
        "pre_idx": cp.asarray(dcn_pre),    # Which DCN provides feedback
        "post_idx": cp.asarray(dcn_post),  # Which IO receives the feedback
    }

    return graph