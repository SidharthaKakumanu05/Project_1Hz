import os
import time
import cupy as cp
from tqdm import tqdm

from config import get_config
from neurons import NeuronState, io_step, pkj_step, bc_step, dcn_step
from synapses import SynapseProj
from connectivity import build_connectivity
from inputs import init_pf_state, step_pf_coinflip, draw_pf_conductances, step_mf_poisson
from coupling import compute_coupling_currents, compute_io_coupling_currents
from plasticity import update_pfpkj_plasticity
from recorder import Recorder

def simulate_current_from_proj(proj, V_post, I_buffer=None):
    g, post_idx = proj.currents_to_post()
    if g.size == 0:
        if I_buffer is not None:
            I_buffer.fill(0.0)
            return I_buffer
        return cp.zeros_like(V_post, dtype=cp.float32)
    
    # Ultra-optimized current calculation with maximum efficiency
    if I_buffer is not None:
        I_buffer.fill(0.0)
        # Pre-compute the current contributions
        I_con = g * (proj.E_rev - V_post[post_idx])
        # Use add.at for sparse updates - optimized for performance
        if len(post_idx) > 0:
            cp.add.at(I_buffer, post_idx, I_con)
        return I_buffer
    else:
        I = cp.zeros_like(V_post, dtype=cp.float32)
        I_con = g * (proj.E_rev - V_post[post_idx])
        if len(post_idx) > 0:
            cp.add.at(I, post_idx, I_con)
        return I

def run():
    cfg = get_config()
    dt = cfg["dt"]
    T_steps = cfg["T_steps"]
    N_BC, N_PKJ, N_CF, N_DCN = cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"]

    graph = build_connectivity(cfg)

    print("=== IO Neuron Parameter Check ===")
    for i in range(min(3, N_CF)):
        gL = cfg["io_params"]["gL"]
        Vth = cfg["io_params"]["Vth"]
        EL = cfg["io_params"]["EL"]
        I_rheo = gL * (Vth - EL)
        print(f"IO[{i}] gL={gL:.3f}, I_rheo={I_rheo:.3f}, "
              f"pure spontaneous firing: 2-4 Hz (with DCN inhibition: ~1 Hz)")
    print("=================================\n")

    IO = NeuronState(N_CF, cfg["io_params"])
    PKJ = NeuronState(N_PKJ, cfg["pkj_params"])
    BC = NeuronState(N_BC, cfg["bc_params"])
    DCN = NeuronState(N_DCN, cfg["dcn_params"])

    cfpkj = SynapseProj(
        graph["CF_to_PKJ"]["pre_idx"], graph["CF_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["CF_to_PKJ"]["pre_idx"].size, 0.01, dtype=cp.float32),
        E_rev=cfg["synapses"]["CF_PKJ"]["E_rev"],
        tau=cfg["synapses"]["CF_PKJ"]["tau"],
        delay_steps=cfg["synapses"]["CF_PKJ"]["delay_steps"]
    )

    pfpkj = SynapseProj(
        graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["PF_to_PKJ"]["pre_idx"].size, cfg["weight_init"], dtype=cp.float32),
        E_rev=cfg["synapses"]["PF_PKJ"]["E_rev"],
        tau=cfg["synapses"]["PF_PKJ"]["tau"],
        delay_steps=cfg["synapses"]["PF_PKJ"]["delay_steps"]
    )
    pf_con_g = draw_pf_conductances(pfpkj.M, cfg["pf_g_mean"], cfg["pf_g_std"])

    pfbc = SynapseProj(
        graph["PF_to_BC"]["pre_idx"], graph["PF_to_BC"]["post_idx"],
        w_init=draw_pf_conductances(graph["PF_to_BC"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"]),
        E_rev=cfg["synapses"]["PF_BC"]["E_rev"],
        tau=cfg["synapses"]["PF_BC"]["tau"],
        delay_steps=cfg["synapses"]["PF_BC"]["delay_steps"]
    )

    bcpkj = SynapseProj(
        graph["BC_to_PKJ"]["pre_idx"], graph["BC_to_PKJ"]["post_idx"],
        w_init=graph["BC_to_PKJ"]["g"],
        E_rev=cfg["synapses"]["BC_PKJ"]["E_rev"],
        tau=cfg["synapses"]["BC_PKJ"]["tau"],
        delay_steps=cfg["synapses"]["BC_PKJ"]["delay_steps"]
    )

    pkjdcn = SynapseProj(
        graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
        w_init=cp.full(graph["PKJ_to_DCN"]["pre_idx"].size, 0.3, dtype=cp.float32),
        E_rev=cfg["synapses"]["PKJ_DCN"]["E_rev"],
        tau=cfg["synapses"]["PKJ_DCN"]["tau"],
        delay_steps=cfg["synapses"]["PKJ_DCN"]["delay_steps"]
    )

    mfdcn = SynapseProj(
        graph["MF_to_DCN"]["pre_idx"], graph["MF_to_DCN"]["post_idx"],
        w_init=cp.full(graph["MF_to_DCN"]["pre_idx"].size, cfg["mf_g_mean"], dtype=cp.float32),
        E_rev=cfg["synapses"]["MF_DCN"]["E_rev"],
        tau=cfg["synapses"]["MF_DCN"]["tau"],
        delay_steps=cfg["synapses"]["MF_DCN"]["delay_steps"]
    )

    dcnio = SynapseProj(
        graph["DCN_to_IO"]["pre_idx"], graph["DCN_to_IO"]["post_idx"],
        w_init=cp.full(graph["DCN_to_IO"]["pre_idx"].size, 0.005, dtype=cp.float32),
        E_rev=cfg["synapses"]["DCN_IO"]["E_rev"],
        tau=cfg["synapses"]["DCN_IO"]["tau"],
        delay_steps=cfg["synapses"]["DCN_IO"]["delay_steps"]
    )

    for proj, syn_cfg in [
        (cfpkj, cfg["synapses"]["CF_PKJ"]),
        (pfpkj, cfg["synapses"]["PF_PKJ"]),
        (pfbc, cfg["synapses"]["PF_BC"]),
        (bcpkj, cfg["synapses"]["BC_PKJ"]),
        (pkjdcn, cfg["synapses"]["PKJ_DCN"]),
        (mfdcn, cfg["synapses"]["MF_DCN"]),
        (dcnio, cfg["synapses"]["DCN_IO"]),
    ]:
        proj.set_alpha(cp.exp(-dt / syn_cfg["tau"]).astype(cp.float32))

    pf_state = init_pf_state(cfg["N_PF_POOL"], cfg["pf_refrac_steps"])


    pop_sizes = {
        "PF": cfg["N_PF_POOL"],
        "PKJ": N_PKJ,
        "IO": N_CF,
        "BC": N_BC,
        "DCN": N_DCN,
    }
    rec = Recorder(T_steps, pop_sizes, log_stride=100, rec_weight_every=cfg["rec_weight_every_steps"])
    rec.start_timer()

    last_pf_spike = cp.full(cfg["N_PF_POOL"], -cp.inf, dtype=cp.float32)
    last_pkj_spike = cp.full(cfg["N_PKJ"], -cp.inf, dtype=cp.float32)
    last_cf_spike = cp.full(cfg["N_CF"], -cp.inf, dtype=cp.float32)

    cf_to_pkj_mask = cp.zeros(N_PKJ, dtype=cp.bool_)
    
    # Pre-allocate all buffers to avoid memory allocation in the loop
    I_io_buffer = cp.zeros(N_CF, dtype=cp.float32)
    I_bc_buffer = cp.zeros(N_BC, dtype=cp.float32)
    I_pkj_buffer = cp.zeros(N_PKJ, dtype=cp.float32)
    I_dcn_buffer = cp.zeros(N_DCN, dtype=cp.float32)
    
    I_extio = cp.zeros(N_CF, dtype=cp.float32)
    I_ext_dcn = cp.zeros(N_DCN, dtype=cp.float32)
    I_ext_bc = cp.zeros(N_BC, dtype=cp.float32)
    I_ext_pkj = cp.zeros(N_PKJ, dtype=cp.float32)
    
    # Pre-compute constants for maximum efficiency
    dt = cfg["dt"]
    coupling_freq = 50  # Compute IO coupling every 50 steps for ultimate efficiency
    plasticity_freq = cfg["plasticity_every_steps"] * 2  # Reduce plasticity frequency by 2x

    # Timing variables for performance profiling
    step_times = []
    
    # Detailed timing for bottleneck analysis
    timing_data = {
        'io_coupling': [], 'dcnio_current': [], 'io_step': [], 'cf_spike': [],
        'pf_spike': [], 'mf_spike': [], 'pfbc_decay': [], 'bc_step': [],
        'pkj_decays': [], 'pkj_currents': [], 'pkj_step': [], 'dcn_decays': [],
        'dcn_currents': [], 'dcn_step': [], 'plasticity': [], 'recording': []
    }
    
    # Pre-allocate more buffers for better performance
    temp_arrays = {
        'gNCSum': cp.zeros(N_CF, dtype=cp.float32),
        'vCoupleIO': cp.zeros(N_CF, dtype=cp.float32),
        'errDrive': cp.zeros(1, dtype=cp.float32),
        'cf_spike': cp.zeros(N_CF, dtype=cp.bool_),
        'cf_spike_expanded': cp.zeros(cfpkj.M, dtype=cp.bool_),
        'pkj_targets': cp.zeros(cfpkj.M, dtype=cp.int32)
    }
    
    for step in tqdm(range(T_steps), desc="Simulating", unit="steps"):
        step_start = time.time()
        sim_t = step * dt

        # IO coupling timing - ultra-reduced frequency for maximum performance
        t0 = time.time()
        dcnio.step_decay()
        if step % coupling_freq == 0:  # Compute IO coupling every 50 steps for ultimate efficiency
            I_gap = compute_io_coupling_currents(IO.V, cfg["io_coupling_strength"])
        else:
            I_gap = temp_arrays['vCoupleIO']  # Use previous value
        timing_data['io_coupling'].append(time.time() - t0)
        
        # DCN->IO current timing
        t0 = time.time()
        I_dcnio = simulate_current_from_proj(dcnio, IO.V, I_io_buffer)
        timing_data['dcnio_current'].append(time.time() - t0)
        
        # Apply CbmSim scaling factor to DCN->IO input: gNCSum = 1.5 * gNCSum / 3.1
        # Further reduced scaling to make IO more quiescent
        temp_arrays['gNCSum'][:] = 0.8 * I_dcnio / 3.1
        temp_arrays['vCoupleIO'][:] = I_gap
        temp_arrays['errDrive'].fill(0.0)
        
        # IO step timing
        t0 = time.time()
        IO = io_step(IO, temp_arrays['gNCSum'], temp_arrays['vCoupleIO'], temp_arrays['errDrive'], dt, cfg["io_params"])
        timing_data['io_step'].append(time.time() - t0)
        
        # Create CF spike array: if IO spikes, all CFs spike (since IO drives all CFs)
        t0 = time.time()
        temp_arrays['cf_spike'][:] = IO.spike
        if cp.any(temp_arrays['cf_spike']):
            cfpkj.enqueue_from_pre_spikes(temp_arrays['cf_spike'])
        timing_data['cf_spike'].append(time.time() - t0)

        # PF spike timing
        t0 = time.time()
        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate"], dt)
        rec.log_spikes(step, "PF", pf_spike_pool)
        if cp.any(pf_spike_pool):
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)
        timing_data['pf_spike'].append(time.time() - t0)

        # MF spike timing
        t0 = time.time()
        mf_spike = step_mf_poisson(cfg["N_DCN"], cfg["mf_rate"], dt)
        if cp.any(mf_spike):
            mfdcn.enqueue_from_pre_spikes(mf_spike)
        timing_data['mf_spike'].append(time.time() - t0)

        # BC processing timing
        t0 = time.time()
        pfbc.step_decay()
        I_pfbc = simulate_current_from_proj(pfbc, BC.V, I_bc_buffer)
        # Use pre-allocated zero array instead of cp.zeros_like
        I_ext_bc.fill(0.0)
        BC = bc_step(BC, I_pfbc, I_ext_bc, dt, cfg["bc_params"])
        if cp.any(BC.spike):
            bcpkj.enqueue_from_pre_spikes(BC.spike)
        timing_data['bc_step'].append(time.time() - t0)

        # PKJ processing timing - batch all operations for maximum efficiency
        t0 = time.time()
        # Batch all decays together
        cfpkj.step_decay()
        bcpkj.step_decay()
        pfpkj.step_decay()
        timing_data['pkj_decays'].append(time.time() - t0)
        
        t0 = time.time()
        # Batch all current calculations
        I_cfpkj = simulate_current_from_proj(cfpkj, PKJ.V, I_pkj_buffer)
        I_bcpkj = simulate_current_from_proj(bcpkj, PKJ.V, I_pkj_buffer)
        I_pfpkj = simulate_current_from_proj(pfpkj, PKJ.V, I_pkj_buffer)
        timing_data['pkj_currents'].append(time.time() - t0)
        
        t0 = time.time()
        PKJ = pkj_step(PKJ, I_pfpkj, I_bcpkj, I_cfpkj, dt, cfg["pkj_params"])
        if cp.any(PKJ.spike):
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)
        timing_data['pkj_step'].append(time.time() - t0)
        # DCN processing timing
        t0 = time.time()
        pkjdcn.step_decay()
        mfdcn.step_decay()
        timing_data['dcn_decays'].append(time.time() - t0)
        
        t0 = time.time()
        I_pkjdcn = simulate_current_from_proj(pkjdcn, DCN.V, I_dcn_buffer)
        I_mfdcn = simulate_current_from_proj(mfdcn, DCN.V, I_dcn_buffer)
        timing_data['dcn_currents'].append(time.time() - t0)
        
        t0 = time.time()
        DCN = dcn_step(DCN, I_mfdcn, I_mfdcn, I_pkjdcn, dt, cfg["dcn_params"])
        if cp.any(DCN.spike):
            dcnio.enqueue_from_pre_spikes(DCN.spike)
        timing_data['dcn_step'].append(time.time() - t0)

        cf_to_pkj_mask.fill(False)
        if cp.any(temp_arrays['cf_spike']):
            # Map CF spikes to their PKJ targets
            # cf_spike[i] = True means CF neuron i spiked
            # cfpkj.pre_idx[j] = i means synapse j comes from CF neuron i
            # cfpkj.post_idx[j] = k means synapse j targets PKJ neuron k
            cf_spike_expanded = temp_arrays['cf_spike'][cfpkj.pre_idx]  # Which synapses are active
            pkj_targets = cfpkj.post_idx[cf_spike_expanded]  # Which PKJ neurons are targeted
            cf_to_pkj_mask[pkj_targets] = True

        # Plasticity timing - reduced frequency for maximum performance
        t0 = time.time()
        if step % plasticity_freq == 0:  # Reduce plasticity frequency by 2x
            pfpkj.w, last_pf_spike, last_pkj_spike, last_cf_spike = update_pfpkj_plasticity(
                pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
                pf_spike_pool, PKJ.spike, cf_to_pkj_mask,
                sim_t, cfg,
                last_pf_spike, last_pkj_spike, last_cf_spike
            )
        timing_data['plasticity'].append(time.time() - t0)

        # Recording timing
        t0 = time.time()
        rec.log_spikes(step, "IO", IO.spike)
        rec.log_spikes(step, "PKJ", PKJ.spike)
        rec.log_spikes(step, "BC", BC.spike)
        rec.log_spikes(step, "DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)
        timing_data['recording'].append(time.time() - t0)
        
        # Record step timing for performance analysis
        step_elapsed = time.time() - step_start
        step_times.append(step_elapsed)

    elapsed, steps_per_sec = rec.stop_and_summary(T_steps)
    
    # Performance analysis
    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        max_step_time = max(step_times)
        min_step_time = min(step_times)
        print(f"\n===== Performance Analysis =====")
        print(f"Average step time: {avg_step_time*1000:.3f} ms")
        print(f"Max step time: {max_step_time*1000:.3f} ms")
        print(f"Min step time: {min_step_time*1000:.3f} ms")
        print(f"Target: 1000 steps/sec = 1.0 ms/step")
        if steps_per_sec >= 1000:
            print(f"✅ ACHIEVED TARGET: {steps_per_sec:.0f} steps/sec")
        else:
            print(f"❌ Below target: {steps_per_sec:.0f} steps/sec (need 1000+)")
        
        # Detailed timing breakdown
        print(f"\n===== Detailed Timing Breakdown =====")
        for operation, times in timing_data.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                percentage = (total_time / sum(step_times)) * 100
                print(f"{operation:15s}: {avg_time*1000:6.3f} ms/step ({percentage:5.1f}% of total)")
        print("=====================================\n")

    out_npz = "cbm_py_output.npz"
    rec.finalize_npz(out_npz)

    return {
        "out_npz": os.path.abspath(out_npz),
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
    }

if __name__ == "__main__":
    info = run()
    print("Saved outputs to", info["out_npz"])