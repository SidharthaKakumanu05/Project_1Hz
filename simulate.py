import os
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
    g, post_idx = proj.currents_to_post()
    I_con = g * (proj.E_rev - V_post[post_idx])
    I = cp.zeros_like(V_post, dtype=cp.float32)
    cp.add.at(I, post_idx, I_con)
    return I


def run():
    cfg = get_config()
    dt = cfg["dt"]
    T_steps = cfg["T_steps"]

    graph = build_connectivity(cfg)

    # --- Neuron populations ---
    IO = NeuronState(cfg["N_CF"], cfg["lif_IO"])
    PKJ = NeuronState(cfg["N_PKJ"], cfg["lif_PKJ"])
    BC = NeuronState(cfg["N_BC"], cfg["lif_BC"])
    DCN = NeuronState(cfg["N_DCN"], cfg["lif_DCN"])

    # --- Synapse Projections ---
    cfpkj = SynapseProj(
        graph["CF_to_PKJ"]["pre_idx"], graph["CF_to_PKJ"]["post_idx"],
        w_init=cp.full(cfg["N_CF"], 8e-9, dtype=cp.float32),
        E_rev=cfg["syn_CF_PKJ"]["E_rev"],
        tau=cfg["syn_CF_PKJ"]["tau"],
        delay_steps=cfg["syn_CF_PKJ"]["delay_steps"],
    )

    pfpkj = SynapseProj(
        graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["PF_to_PKJ"]["pre_idx"].size, cfg["w_pfpkj_init"], dtype=cp.float32),
        E_rev=cfg["syn_PF_PKJ"]["E_rev"],
        tau=cfg["syn_PF_PKJ"]["tau"],
        delay_steps=cfg["syn_PF_PKJ"]["delay_steps"],
    )
    pf_con_g = draw_pf_conductances(
        pfpkj.M, cfg["pf_g_mean"], cfg["pf_g_std"]
    )

    pfbc = SynapseProj(
        graph["PF_to_BC"]["pre_idx"], graph["PF_to_BC"]["post_idx"],
        w_init=draw_pf_conductances(graph["PF_to_BC"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"]),
        E_rev=cfg["syn_PF_BC"]["E_rev"],
        tau=cfg["syn_PF_BC"]["tau"],
        delay_steps=cfg["syn_PF_BC"]["delay_steps"],
    )

    bcpkj = SynapseProj(
        graph["BC_to_PKJ"]["pre_idx"], graph["BC_to_PKJ"]["post_idx"],
        w_init=graph["BC_to_PKJ"]["g"],
        E_rev=cfg["syn_BC_PKJ"]["E_rev"],
        tau=cfg["syn_BC_PKJ"]["tau"],
        delay_steps=cfg["syn_BC_PKJ"]["delay_steps"],
    )

    pkjdcn = SynapseProj(
        graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
        w_init=cp.full(graph["PKJ_to_DCN"]["pre_idx"].size, 9e-9, dtype=cp.float32),
        E_rev=cfg["syn_PKJ_DCN"]["E_rev"],
        tau=cfg["syn_PKJ_DCN"]["tau"],
        delay_steps=cfg["syn_PKJ_DCN"]["delay_steps"],
    )

    mfdcn = SynapseProj(
        graph["MF_to_DCN"]["pre_idx"], graph["MF_to_DCN"]["post_idx"],
        w_init=cp.full(graph["MF_to_DCN"]["pre_idx"].size, cfg["mf_g_mean"], dtype=cp.float32),
        E_rev=cfg["syn_MF_DCN"]["E_rev"],
        tau=cfg["syn_MF_DCN"]["tau"],
        delay_steps=cfg["syn_MF_DCN"]["delay_steps"],
    )

    # decay factors
    for proj, syn_cfg in [
        (cfpkj, cfg["syn_CF_PKJ"]),
        (pfpkj, cfg["syn_PF_PKJ"]),
        (pfbc, cfg["syn_PF_BC"]),
        (bcpkj, cfg["syn_BC_PKJ"]),
        (pkjdcn, cfg["syn_PKJ_DCN"]),
        (mfdcn, cfg["syn_MF_DCN"]),
    ]:
        proj.set_alpha(cp.exp(-dt / syn_cfg["tau"]).astype(cp.float32))

    pf_state = init_pf_state(cfg["N_PF_POOL"], cfg["pf_refrac_steps"])

    # gap junction pairs for IO
    if cfg["N_CF"] > 1:
        pairs = cp.stack([cp.arange(cfg["N_CF"]), cp.roll(cp.arange(cfg["N_CF"]), -1)], axis=1).astype(cp.int32)
    else:
        pairs = cp.zeros((0, 2), dtype=cp.int32)

    rec = Recorder(cfg)

    traces = None

    for step in tqdm(range(T_steps), desc="Simulating", unit="steps"):
        # IO
        I_gap = apply_ohmic_coupling(IO.V, pairs, cfg["g_gap_IO"])
        I_ext_io = cp.full(IO.V.shape, cfg["io_bias_current"], dtype=cp.float32)
        IO = lif_step(IO, I_syn=I_gap, I_ext=I_ext_io, dt=dt, lif_cfg=cfg["lif_IO"])
        cf_spike = IO.spike.copy()
        if cp.any(cf_spike):
            cfpkj.enqueue_from_pre_spikes(cf_spike)

        # PF/MF
        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate_hz"], dt)
        rec.log_spikes("PF", pf_spike_pool)
        if cp.any(pf_spike_pool):
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)
        mf_spike = step_mf_poisson(cfg["N_PF_POOL"], cfg["mf_rate_hz"], dt)
        if cp.any(mf_spike):
            mfdcn.enqueue_from_pre_spikes(mf_spike)

        # Basket
        pfbc.step_decay()
        I_bc = simulate_current_from_proj(pfbc, cp.zeros(cfg["N_BC"], dtype=cp.float32))
        BC = lif_step(BC, I_syn=I_bc, I_ext=cp.zeros(cfg["N_BC"], dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_BC"])
        if cp.any(BC.spike):
            bcpkj.enqueue_from_pre_spikes(BC.spike, scale=graph["BC_to_PKJ"]["g"])

        # PKJ
        cfpkj.step_decay()
        bcpkj.step_decay()
        pfpkj.step_decay()
        I_pkj = (
            simulate_current_from_proj(cfpkj, PKJ.V)
            + simulate_current_from_proj(bcpkj, PKJ.V)
            + simulate_current_from_proj(pfpkj, PKJ.V)
        )
        PKJ = lif_step(PKJ, I_syn=I_pkj, I_ext=cp.zeros(cfg["N_PKJ"], dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_PKJ"])

        # DCN
        if cp.any(PKJ.spike):
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)
        pkjdcn.step_decay()
        mfdcn.step_decay()
        I_dcn = simulate_current_from_proj(pkjdcn, DCN.V) + simulate_current_from_proj(mfdcn, DCN.V)
        DCN = lif_step(DCN, I_syn=I_dcn, I_ext=cp.zeros(cfg["N_DCN"], dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_DCN"])

        # --- Plasticity (CF gate + pre/post spike alignment) ---
        # CF -> PKJ mask: which PKJs received a CF spike this step?
        cf_to_pkj_mask = cp.zeros(cfg["N_PKJ"], dtype=bool)
        if cp.any(cf_spike):
            # cf_spike is shape (N_CF,), cfpkj.post_idx is shape (N_CF,)
            pkj_targets = cfpkj.post_idx[cf_spike]     # boolean-index directly
            cf_to_pkj_mask[pkj_targets] = True

        # Pre spikes must be aligned to the synapse list the rule sees.
        # pf_spike_pool is over PF pool indices (0..N_PF_POOL-1).
        # The plasticity function uses (pre_idx, post_idx) to map; give it
        # the *subset* of PFs that actually have synapses by indexing with pre_idx.
        pf_pre_active = pf_spike_pool[pfpkj.pre_idx]   # shape (M,) for M synapses

        # Run rule
        pfpkj.w, traces = update_pfpkj_plasticity(
            pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
            pf_pre_active,                 # <-- use synapse-aligned presyn spikes
            PKJ.spike,                     # postsyn spikes (N_PKJ,)
            cf_to_pkj_mask,                # CF gate per PKJ
            dt, cfg, traces=traces
        )

        # Record
        rec.log_spikes("IO", IO.spike)
        rec.log_spikes("PKJ", PKJ.spike)
        rec.log_spikes("BC", BC.spike)
        rec.log_spikes("DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)

    out_npz = "cbm_py_output.npz"
    rec.finalize_npz(out_npz)
    return {"out_npz": os.path.abspath(out_npz)}


if __name__ == "__main__":
    info = run()
    print("Saved outputs to", info["out_npz"])