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
    
    I_con = g * (proj.E_rev - V_post[post_idx])
    if I_buffer is not None:
        I_buffer.fill(0.0)
        cp.add.at(I_buffer, post_idx, I_con)
        return I_buffer
    else:
        I = cp.zeros_like(V_post, dtype=cp.float32)
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
    rec = Recorder(T_steps, pop_sizes, log_stride=50, rec_weight_every=cfg["rec_weight_every_steps"])
    rec.start_timer()

    last_pf_spike = cp.full(cfg["N_PF_POOL"], -cp.inf, dtype=cp.float32)
    last_pkj_spike = cp.full(cfg["N_PKJ"], -cp.inf, dtype=cp.float32)
    last_cf_spike = cp.full(cfg["N_CF"], -cp.inf, dtype=cp.float32)

    cf_to_pkj_mask = cp.zeros(N_PKJ, dtype=cp.bool_)
    
    I_io_buffer = cp.zeros(N_CF, dtype=cp.float32)
    I_bc_buffer = cp.zeros(N_BC, dtype=cp.float32)
    
    I_extio = cp.zeros(N_CF, dtype=cp.float32)
    I_ext_dcn = cp.zeros(N_DCN, dtype=cp.float32)
    I_ext_bc = cp.zeros(N_BC, dtype=cp.float32)
    I_ext_pkj = cp.zeros(N_PKJ, dtype=cp.float32)

    for step in tqdm(range(T_steps), desc="Simulating", unit="steps"):
        sim_t = step * dt

        dcnio.step_decay()
        I_gap = compute_io_coupling_currents(IO.V, cfg["io_coupling_strength"])
        I_dcnio = simulate_current_from_proj(dcnio, IO.V, I_io_buffer)
        
        # Apply CbmSim scaling factor to DCN->IO input: gNCSum = 1.5 * gNCSum / 3.1
        # Further reduced scaling to make IO more quiescent
        gNCSum = 0.8 * I_dcnio / 3.1
        vCoupleIO = I_gap
        errDrive = cp.zeros(1, dtype=cp.float32)
        
        IO = io_step(IO, gNCSum, vCoupleIO, errDrive, dt, cfg["io_params"])
        
        # Create CF spike array: if IO spikes, all CFs spike (since IO drives all CFs)
        cf_spike = cp.full(N_CF, IO.spike, dtype=cp.bool_)
        if cp.any(cf_spike):
            cfpkj.enqueue_from_pre_spikes(cf_spike)

        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate"], dt)
        rec.log_spikes(step, "PF", pf_spike_pool)
        if cp.any(pf_spike_pool):
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)

        mf_spike = step_mf_poisson(cfg["N_DCN"], cfg["mf_rate"], dt)
        if cp.any(mf_spike):
            mfdcn.enqueue_from_pre_spikes(mf_spike)

        pfbc.step_decay()
        I_pfbc = simulate_current_from_proj(pfbc, BC.V, I_bc_buffer)
        BC = bc_step(BC, I_pfbc, cp.zeros_like(I_pfbc), dt, cfg["bc_params"])
        if cp.any(BC.spike):
            bcpkj.enqueue_from_pre_spikes(BC.spike)

        cfpkj.step_decay()
        bcpkj.step_decay()
        pfpkj.step_decay()
        I_cfpkj = simulate_current_from_proj(cfpkj, PKJ.V)
        I_bcpkj = simulate_current_from_proj(bcpkj, PKJ.V)
        I_pfpkj = simulate_current_from_proj(pfpkj, PKJ.V)
        PKJ = pkj_step(PKJ, I_pfpkj, I_bcpkj, I_cfpkj, dt, cfg["pkj_params"])

        if cp.any(PKJ.spike):
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)
        pkjdcn.step_decay()
        mfdcn.step_decay()
        I_pkjdcn = simulate_current_from_proj(pkjdcn, DCN.V)
        I_mfdcn = simulate_current_from_proj(mfdcn, DCN.V)
        DCN = dcn_step(DCN, I_mfdcn, I_mfdcn, I_pkjdcn, dt, cfg["dcn_params"])
        
        if cp.any(DCN.spike):
            dcnio.enqueue_from_pre_spikes(DCN.spike)

        cf_to_pkj_mask.fill(False)
        if cp.any(cf_spike):
            # Map CF spikes to their PKJ targets
            # cf_spike[i] = True means CF neuron i spiked
            # cfpkj.pre_idx[j] = i means synapse j comes from CF neuron i
            # cfpkj.post_idx[j] = k means synapse j targets PKJ neuron k
            cf_spike_expanded = cf_spike[cfpkj.pre_idx]  # Which synapses are active
            pkj_targets = cfpkj.post_idx[cf_spike_expanded]  # Which PKJ neurons are targeted
            cf_to_pkj_mask[pkj_targets] = True

        if step % cfg["plasticity_every_steps"] == 0:
            pfpkj.w, last_pf_spike, last_pkj_spike, last_cf_spike = update_pfpkj_plasticity(
                pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
                pf_spike_pool, PKJ.spike, cf_to_pkj_mask,
                sim_t, cfg,
                last_pf_spike, last_pkj_spike, last_cf_spike
            )

        rec.log_spikes(step, "IO", IO.spike)
        rec.log_spikes(step, "PKJ", PKJ.spike)
        rec.log_spikes(step, "BC", BC.spike)
        rec.log_spikes(step, "DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)

    elapsed, steps_per_sec = rec.stop_and_summary(T_steps)

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