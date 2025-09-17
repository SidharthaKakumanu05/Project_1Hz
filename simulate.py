import os
import numpy as np
import cupy as cp
from tqdm import tqdm

from config import get_config
from neurons import NeuronState, lif_step
from synapses import SynapseProj
from connectivity import build_connectivity
from inputs import init_pf_state, step_pf_coinflip, draw_pf_conductances, step_mf_poisson
from coupling import apply_ohmic_coupling
from plasticity import update_pfpkj_plasticity
from recorder import Recorder


def simulate_current_from_proj(proj, V_post):
    """
    Convert synapse conductances to currents on postsynaptic neurons.
    Each synapse contributes:
        I = g * (E_rev - V_post[post_idx])
    summed over all synapses onto each postsynaptic neuron.
    """
    g, post_idx = proj.currents_to_post()  # g: [M], post_idx: [M]
    I = cp.zeros_like(V_post, dtype=cp.float32)
    I_con = g * (proj.E_rev - V_post[post_idx])
    cp.scatter_add(I, post_idx, I_con)  # accumulate per postsynaptic neuron
    return I

def run():
    cfg = get_config()
    cp.random.seed(cfg["seed"])   # global RNG seed

    dt      = cfg["dt"]
    T_steps = cfg["T_steps"]
    N_BC    = cfg["N_BC"]
    N_PKJ   = cfg["N_PKJ"]
    N_CF    = cfg["N_CF"]
    N_DCN   = cfg["N_DCN"]
    N_PF    = cfg["N_PF_POOL"]

    # -------- populations --------
    IO  = NeuronState(N_CF,  cfg["lif_IO"])
    PKJ = NeuronState(N_PKJ, cfg["lif_PKJ"])
    BC  = NeuronState(N_BC,  cfg["lif_BC"])
    DCN = NeuronState(N_DCN, cfg["lif_DCN"])

    # -------- connectivity --------
    graph = build_connectivity(cfg)

    # CF → PKJ
    cfpkj = SynapseProj(graph["CF_to_PKJ"]["pre_idx"], graph["CF_to_PKJ"]["post_idx"],
                        w_init=cp.full(graph["CF_to_PKJ"]["pre_idx"].size, 8e-9, dtype=cp.float32),
                        E_rev=cfg["syn_CF_PKJ"]["E_rev"], tau=cfg["syn_CF_PKJ"]["tau"],
                        delay_steps=cfg["syn_CF_PKJ"]["delay_steps"])

    # PF → PKJ
    pfpkj = SynapseProj(graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
                        w_init=cp.full(graph["PF_to_PKJ"]["pre_idx"].size, cfg["w_pfpkj_init"], dtype=cp.float32),
                        E_rev=cfg["syn_PF_PKJ"]["E_rev"], tau=cfg["syn_PF_PKJ"]["tau"],
                        delay_steps=cfg["syn_PF_PKJ"]["delay_steps"])
    pf_con_g = draw_pf_conductances(pfpkj.w.size, cfg["pf_g_mean"], cfg["pf_g_std"])

    # PF → BC
    pfbc = SynapseProj(graph["PF_to_BC"]["pre_idx"], graph["PF_to_BC"]["post_idx"],
                       w_init=draw_pf_conductances(graph["PF_to_BC"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"]),
                       E_rev=cfg["syn_PF_BC"]["E_rev"], tau=cfg["syn_PF_BC"]["tau"],
                       delay_steps=cfg["syn_PF_BC"]["delay_steps"])

    # BC → PKJ
    bcpkj = SynapseProj(graph["BC_to_PKJ"]["pre_idx"], graph["BC_to_PKJ"]["post_idx"],
                        w_init=cp.full(graph["BC_to_PKJ"]["pre_idx"].size, 4e-9, dtype=cp.float32),
                        E_rev=cfg["syn_BC_PKJ"]["E_rev"], tau=cfg["syn_BC_PKJ"]["tau"],
                        delay_steps=cfg["syn_BC_PKJ"]["delay_steps"])

    # PKJ → DCN
    pkjdcn = SynapseProj(graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
                         w_init=cp.full(graph["PKJ_to_DCN"]["pre_idx"].size, 9e-9, dtype=cp.float32),
                         E_rev=cfg["syn_PKJ_DCN"]["E_rev"], tau=cfg["syn_PKJ_DCN"]["tau"],
                         delay_steps=cfg["syn_PKJ_DCN"]["delay_steps"])

    # MF → DCN
    mfdcn = SynapseProj(graph["MF_to_DCN"]["pre_idx"], graph["MF_to_DCN"]["post_idx"],
                        w_init=cp.full(graph["MF_to_DCN"]["pre_idx"].size, cfg["mf_g_mean"], dtype=cp.float32),
                        E_rev=cfg["syn_MF_DCN"]["E_rev"], tau=cfg["syn_MF_DCN"]["tau"],
                        delay_steps=cfg["syn_MF_DCN"]["delay_steps"])

    # set decay alphas
    for proj, syn_cfg in [
        (cfpkj, cfg["syn_CF_PKJ"]), (pfpkj, cfg["syn_PF_PKJ"]),
        (pfbc,  cfg["syn_PF_BC"]),  (bcpkj, cfg["syn_BC_PKJ"]),
        (pkjdcn,cfg["syn_PKJ_DCN"]), (mfdcn, cfg["syn_MF_DCN"])
    ]:
        proj.set_alpha(cp.exp(cp.float32(-dt / syn_cfg["tau"])))

    # PF pool
    pf_state = init_pf_state(N_PF, cfg["pf_refrac_steps"])

    # IO coupling pairs (ring)
    if N_CF > 1:
        pairs = cp.stack([cp.arange(N_CF, dtype=cp.int32),
                          cp.roll(cp.arange(N_CF, dtype=cp.int32), -1)], axis=1)
    else:
        pairs = cp.zeros((0, 2), dtype=cp.int32)

    rec = Recorder(cfg)
    traces = None  # for plasticity

    # CF→PKJ 1:1 map
    cf2pkj_map = cp.full(N_CF, -1, dtype=cp.int32)
    cf2pkj_map[cfpkj.pre_idx] = cfpkj.post_idx

    # ---------------- main loop ----------------
    for step in tqdm(range(T_steps), desc="Simulating", unit="s"):
        sim_t = step * dt

        # IO
        I_gap = apply_ohmic_coupling(IO.V, pairs, cfg["g_gap_IO"])
        I_ext_io = cp.full(IO.V.size, cfg["io_bias_current"], dtype=cp.float32)
        IO = lif_step(IO, I_syn=I_gap, I_ext=I_ext_io, dt=dt, lif_cfg=cfg["lif_IO"])
        cf_spike = IO.spike.copy()
        if cf_spike.any():
            cfpkj.enqueue_from_pre_spikes(cf_spike)

        # PF / MF
        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate_hz"], dt)
        rec.log_spikes("PF", pf_spike_pool)
        if pf_spike_pool.any():
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)

        mf_spike = step_mf_poisson(N_DCN, cfg["mf_rate_hz"], dt)
        if mf_spike.any():
            mfdcn.enqueue_from_pre_spikes(mf_spike)

        # BC
        pfbc.step_decay()
        I_bc = simulate_current_from_proj(pfbc, BC.V)
        BC = lif_step(BC, I_syn=I_bc, I_ext=cp.zeros(N_BC, dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_BC"])
        if BC.spike.any():
            bcpkj.enqueue_from_pre_spikes(BC.spike)

        # PKJ
        cfpkj.step_decay(); bcpkj.step_decay(); pfpkj.step_decay()
        I_pkj = simulate_current_from_proj(cfpkj, PKJ.V)
        I_pkj += simulate_current_from_proj(bcpkj, PKJ.V)
        I_pkj += simulate_current_from_proj(pfpkj, PKJ.V)
        PKJ = lif_step(PKJ, I_syn=I_pkj, I_ext=cp.zeros(N_PKJ, dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_PKJ"])

        # DCN
        if PKJ.spike.any():
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)
        pkjdcn.step_decay(); mfdcn.step_decay()
        I_dcn = simulate_current_from_proj(pkjdcn, DCN.V)
        I_dcn += simulate_current_from_proj(mfdcn, DCN.V)
        DCN = lif_step(DCN, I_syn=I_dcn, I_ext=cp.zeros(N_DCN, dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_DCN"])

        # Plasticity
        cf_to_pkj_mask = cp.zeros(N_PKJ, dtype=bool)
        if cf_spike.any():
            cf_idx = cp.where(cf_spike)[0]
            pkj_targets = cf2pkj_map[cf_idx]
            cf_to_pkj_mask[pkj_targets] = True

        pfpkj.w, traces = update_pfpkj_plasticity(
            pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
            pf_spike_pool, PKJ.spike, cf_to_pkj_mask,
            dt, cfg, traces=traces
        )

        # Record
        rec.log_spikes("IO", IO.spike)
        rec.log_spikes("PKJ", PKJ.spike)
        rec.log_spikes("BC", BC.spike)
        rec.log_spikes("DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)

        if step % int(1.0 / dt) == 0 and step > 0:
            tqdm.write(f"Reached {sim_t:.1f} s simulated")

    # save
    out_npz = "cbm_py_output.npz"
    rec.finalize_npz(out_npz)
    return {"out_npz": os.path.abspath(out_npz)}


if __name__ == "__main__":
    info = run()
    print("Saved outputs to", info["out_npz"])