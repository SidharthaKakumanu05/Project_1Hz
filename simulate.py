import os
import numpy as np
from tqdm import tqdm
from config import get_config
from neurons import NeuronState, lif_step
from synapses import SynapseProj
from connectivity import build_connectivity
from inputs import init_pf_state, step_pf_coinflip, draw_pf_conductances, step_mf_poisson
from coupling import apply_ohmic_coupling
from plasticity import update_pfpkj_plasticity
from recorder import Recorder

def run():
    cfg = get_config()
    rng = np.random.default_rng(cfg["seed"])
    dt = cfg["dt"]; T_steps = cfg["T_steps"]
    N_BC, N_PKJ, N_CF, N_DCN = cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"]
    graph = build_connectivity(cfg, rng)

    IO  = NeuronState(N_CF, cfg["lif_IO"], rng)
    PKJ = NeuronState(N_PKJ, cfg["lif_PKJ"], rng)
    BC  = NeuronState(N_BC,  cfg["lif_BC"], rng)
    DCN = NeuronState(N_DCN, cfg["lif_DCN"], rng)

    cfpkj = SynapseProj(graph["CF_to_PKJ"]["pre_idx"], graph["CF_to_PKJ"]["post_idx"],
                        w_init=np.full(N_CF, 8e-9, dtype=np.float32),
                        E_rev=cfg["syn_CF_PKJ"]["E_rev"],
                        tau=cfg["syn_CF_PKJ"]["tau"],
                        delay_steps=cfg["syn_CF_PKJ"]["delay_steps"],
                        rng=rng)

    pfpkj = SynapseProj(graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
                        w_init=np.full(graph["PF_to_PKJ"]["pre_idx"].size, cfg["w_pfpkj_init"], dtype=np.float32),
                        E_rev=cfg["syn_PF_PKJ"]["E_rev"],
                        tau=cfg["syn_PF_PKJ"]["tau"],
                        delay_steps=cfg["syn_PF_PKJ"]["delay_steps"],
                        rng=rng)
    pf_con_g = draw_pf_conductances(pfpkj.M, cfg["pf_g_mean"], cfg["pf_g_std"], rng)

    pfbc = SynapseProj(graph["PF_to_BC"]["pre_idx"], graph["PF_to_BC"]["post_idx"],
                       w_init=draw_pf_conductances(graph["PF_to_BC"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"], rng),
                       E_rev=cfg["syn_PF_BC"]["E_rev"],
                       tau=cfg["syn_PF_BC"]["tau"],
                       delay_steps=cfg["syn_PF_BC"]["delay_steps"],
                       rng=rng)

    bcpkj = SynapseProj(graph["BC_to_PKJ"]["pre_idx"], graph["BC_to_PKJ"]["post_idx"],
                        w_init=np.full(graph["BC_to_PKJ"]["pre_idx"].size, 4e-9, dtype=np.float32),
                        E_rev=cfg["syn_BC_PKJ"]["E_rev"],
                        tau=cfg["syn_BC_PKJ"]["tau"],
                        delay_steps=cfg["syn_BC_PKJ"]["delay_steps"],
                        rng=rng)

    pkjdcn = SynapseProj(graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
                         w_init=np.full(graph["PKJ_to_DCN"]["pre_idx"].size, 9e-9, dtype=np.float32),
                         E_rev=cfg["syn_PKJ_DCN"]["E_rev"],
                         tau=cfg["syn_PKJ_DCN"]["tau"],
                         delay_steps=cfg["syn_PKJ_DCN"]["delay_steps"],
                         rng=rng)

    mfdcn = SynapseProj(graph["MF_to_DCN"]["pre_idx"], graph["MF_to_DCN"]["post_idx"],
                        w_init=np.full(cfg["N_DCN"], cfg["mf_g_mean"], dtype=np.float32),
                        E_rev=cfg["syn_MF_DCN"]["E_rev"],
                        tau=cfg["syn_MF_DCN"]["tau"],
                        delay_steps=cfg["syn_MF_DCN"]["delay_steps"],
                        rng=rng)

    for proj, syn_cfg in [(cfpkj, cfg["syn_CF_PKJ"]), (pfpkj, cfg["syn_PF_PKJ"]), (pfbc, cfg["syn_PF_BC"]),
                          (bcpkj, cfg["syn_BC_PKJ"]), (pkjdcn, cfg["syn_PKJ_DCN"]), (mfdcn, cfg["syn_MF_DCN"])]:
        proj.set_alpha(np.exp(-dt / syn_cfg["tau"]).astype(np.float32))

    pf_state = init_pf_state(cfg["N_PF_POOL"], cfg["pf_refrac_steps"], rng)

    if cfg["N_CF"] > 1:
        pairs = np.stack([np.arange(cfg["N_CF"]), np.roll(np.arange(cfg["N_CF"]), -1)], axis=1).astype(np.int32)
    else:
        pairs = np.zeros((0,2), dtype=np.int32)

    rec = Recorder(cfg)

    def accumulate_current(proj, V_post):
        g, post_idx = proj.currents_to_post()
        I_con = g * (proj.E_rev - V_post[post_idx])
        I = np.zeros_like(V_post, dtype=np.float32)
        np.add.at(I, post_idx, I_con)
        return I
    
    traces = None

    for step in tqdm(range(T_steps), desc="Simulating", unit="s"):
        sim_t = step * dt

        # --- IO ---
        I_gap = apply_ohmic_coupling(IO.V, pairs, cfg["g_gap_IO"])
        I_ext_io = np.full(IO.V.size, cfg["io_bias_current"], dtype=np.float32)
        IO = lif_step(IO, I_syn=I_gap, I_ext=I_ext_io, dt=cfg["dt"], lif_cfg=cfg["lif_IO"])
        cf_spike = IO.spike.copy()
        if np.any(cf_spike):
            cfpkj.enqueue_from_pre_spikes(cf_spike)

        # --- PF/MF ---
        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate_hz"], cfg["dt"], rng)
        rec.log_spikes("PF", pf_spike_pool)
        if np.any(pf_spike_pool):
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)
        mf_spike = step_mf_poisson(cfg["N_DCN"], cfg["mf_rate_hz"], cfg["dt"], rng)
        if np.any(mf_spike):
            mfdcn.enqueue_from_pre_spikes(mf_spike)

        # --- Basket ---
        pfbc.step_decay()
        I_bc = accumulate_current(pfbc, V_post=np.zeros(cfg["N_BC"], dtype=np.float32))
        BC = lif_step(BC, I_syn=I_bc, I_ext=np.zeros(cfg["N_BC"], dtype=np.float32),
                    dt=cfg["dt"], lif_cfg=cfg["lif_BC"])
        if np.any(BC.spike):
            bcpkj.enqueue_from_pre_spikes(BC.spike)

        # --- PKJ ---
        cfpkj.step_decay(); bcpkj.step_decay(); pfpkj.step_decay()
        I_pkj = (accumulate_current(cfpkj, PKJ.V) +
                accumulate_current(bcpkj, PKJ.V) +
                accumulate_current(pfpkj, PKJ.V))
        PKJ = lif_step(PKJ, I_syn=I_pkj, I_ext=np.zeros(cfg["N_PKJ"], dtype=np.float32),
                    dt=cfg["dt"], lif_cfg=cfg["lif_PKJ"])

        # --- DCN ---
        if np.any(PKJ.spike):
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)
        pkjdcn.step_decay(); mfdcn.step_decay()
        I_dcn = accumulate_current(pkjdcn, DCN.V) + accumulate_current(mfdcn, DCN.V)
        DCN = lif_step(DCN, I_syn=I_dcn, I_ext=np.zeros(cfg["N_DCN"], dtype=np.float32),
                    dt=cfg["dt"], lif_cfg=cfg["lif_DCN"])

        # --- Plasticity ---
        cf_to_pkj_mask = np.zeros(cfg["N_PKJ"], dtype=bool)
        if np.any(cf_spike):
            cf_idx = np.where(cf_spike)[0]
            pkj_map = np.empty(cfg["N_CF"], dtype=np.int32)
            pkj_map[:] = -1
            pkj_map[cfpkj.pre_idx] = cfpkj.post_idx
            pkj_targets = pkj_map[cf_idx]
            cf_to_pkj_mask[pkj_targets] = True

        pfpkj.w, traces = update_pfpkj_plasticity(
            pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
            pf_spike_pool, PKJ.spike, cf_to_pkj_mask,
            cfg["dt"], cfg, traces=traces)

        # --- Record ---
        rec.log_spikes("IO", IO.spike)
        rec.log_spikes("PKJ", PKJ.spike)
        rec.log_spikes("BC", BC.spike)
        rec.log_spikes("DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)

        # --- Progress log every second ---
        if step % int(1.0/dt) == 0 and step > 0:
            tqdm.write(f"Reached {sim_t:.1f} s simulated")

    # Finalize once after loop
    out_npz = "cbm_py_output.npz"
    rec.finalize_npz(out_npz)
    return {"out_npz": os.path.abspath(out_npz)}

if __name__ == "__main__":
    info = run()
    print("Saved outputs to", info["out_npz"])
