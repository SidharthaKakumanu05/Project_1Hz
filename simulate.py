import os
import numpy as np                   # only for saving paths/printing
import cupy as cp
from tqdm import tqdm

from config import get_config
from neurons import NeuronState, lif_step
from synapses import SynapseProj
from connectivity import build_connectivity
from inputs import (
    init_pf_state, step_pf_coinflip, draw_pf_conductances, step_mf_poisson
)
from coupling import apply_ohmic_coupling
from plasticity import update_pfpkj_plasticity
from recorder import Recorder


def simulate_current_from_proj(proj: SynapseProj, V_post: cp.ndarray) -> cp.ndarray:
    """
    Convert a synapse projection's conductance to **current** on the postsynaptic population.
    Uses I = g_post * (E_rev - V_post), where g_post is the sum of conductances onto each post idx.
    """
    g_post, post_idx = proj.currents_to_post()  # g summed per postsynaptic index
    # Allocate full-sized current vector
    I = cp.zeros_like(V_post, dtype=cp.float32)
    # Current per post index: g * (Erev - V_post)
    I_con = g_post * (proj.E_rev - V_post[: g_post.size])
    I[: g_post.size] = I_con
    return I


def run():
    cfg = get_config()
    rng = cp.random.default_rng(cfg["seed"])

    dt       = cfg["dt"]
    T_steps  = cfg["T_steps"]
    N_BC     = cfg["N_BC"]
    N_PKJ    = cfg["N_PKJ"]
    N_CF     = cfg["N_CF"]
    N_DCN    = cfg["N_DCN"]
    N_PF     = cfg["N_PF_POOL"]

    # -------- populations --------
    IO  = NeuronState(N_CF,  cfg["lif_IO"],  rng)
    PKJ = NeuronState(N_PKJ, cfg["lif_PKJ"], rng)
    BC  = NeuronState(N_BC,  cfg["lif_BC"],  rng)
    DCN = NeuronState(N_DCN, cfg["lif_DCN"], rng)

    # -------- connectivity --------
    graph = build_connectivity(cfg, rng)

    # CF → PKJ (1:1)
    cfpkj = SynapseProj(
        graph["CF_to_PKJ"]["pre_idx"], graph["CF_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["CF_to_PKJ"]["pre_idx"].size, 8e-9, dtype=cp.float32),
        E_rev=cfg["syn_CF_PKJ"]["E_rev"], tau=cfg["syn_CF_PKJ"]["tau"],
        delay_steps=cfg["syn_CF_PKJ"]["delay_steps"], rng=rng
    )

    # PF → PKJ
    pfpkj = SynapseProj(
        graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["PF_to_PKJ"]["pre_idx"].size, cfg["w_pfpkj_init"], dtype=cp.float32),
        E_rev=cfg["syn_PF_PKJ"]["E_rev"], tau=cfg["syn_PF_PKJ"]["tau"],
        delay_steps=cfg["syn_PF_PKJ"]["delay_steps"], rng=rng
    )
    # per-connection PF conductance scaling (random)
    pf_con_g = draw_pf_conductances(pfpkj.w.size, cfg["pf_g_mean"], cfg["pf_g_std"], rng)

    # PF → BC
    pfbc = SynapseProj(
        graph["PF_to_BC"]["pre_idx"], graph["PF_to_BC"]["post_idx"],
        w_init=draw_pf_conductances(graph["PF_to_BC"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"], rng),
        E_rev=cfg["syn_PF_BC"]["E_rev"], tau=cfg["syn_PF_BC"]["tau"],
        delay_steps=cfg["syn_PF_BC"]["delay_steps"], rng=rng
    )

    # BC → PKJ
    bcpkj = SynapseProj(
        graph["BC_to_PKJ"]["pre_idx"], graph["BC_to_PKJ"]["post_idx"],
        w_init=cp.full(graph["BC_to_PKJ"]["pre_idx"].size, 4e-9, dtype=cp.float32),
        E_rev=cfg["syn_BC_PKJ"]["E_rev"], tau=cfg["syn_BC_PKJ"]["tau"],
        delay_steps=cfg["syn_BC_PKJ"]["delay_steps"], rng=rng
    )

    # PKJ → DCN
    pkjdcn = SynapseProj(
        graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
        w_init=cp.full(graph["PKJ_to_DCN"]["pre_idx"].size, 9e-9, dtype=cp.float32),
        E_rev=cfg["syn_PKJ_DCN"]["E_rev"], tau=cfg["syn_PKJ_DCN"]["tau"],
        delay_steps=cfg["syn_PKJ_DCN"]["delay_steps"], rng=rng
    )

    # MF → DCN
    mfdcn = SynapseProj(
        graph["MF_to_DCN"]["pre_idx"], graph["MF_to_DCN"]["post_idx"],
        w_init=cp.full(graph["MF_to_DCN"]["pre_idx"].size, cfg["mf_g_mean"], dtype=cp.float32),
        E_rev=cfg["syn_MF_DCN"]["E_rev"], tau=cfg["syn_MF_DCN"]["tau"],
        delay_steps=cfg["syn_MF_DCN"]["delay_steps"], rng=rng
    )

    # set synaptic decay alphas
    for proj, syn_cfg in [
        (cfpkj, cfg["syn_CF_PKJ"]), (pfpkj, cfg["syn_PF_PKJ"]),
        (pfbc,  cfg["syn_PF_BC"]),  (bcpkj, cfg["syn_BC_PKJ"]),
        (pkjdcn,cfg["syn_PKJ_DCN"]), (mfdcn, cfg["syn_MF_DCN"])
    ]:
        proj.set_alpha(cp.exp(cp.float32(-dt / syn_cfg["tau"])))

    # PF state
    pf_state = init_pf_state(N_PF, cfg["pf_refrac_steps"], rng)

    # IO gap-junction pairs (ring)
    if N_CF > 1:
        pairs = cp.stack([cp.arange(N_CF, dtype=cp.int32),
                          cp.roll(cp.arange(N_CF, dtype=cp.int32), -1)], axis=1)
    else:
        pairs = cp.zeros((0, 2), dtype=cp.int32)

    # recorder
    rec = Recorder(cfg)

    # plasticity traces holder
    traces = None

    # precompute CF→PKJ mapping (1:1)
    # Build map of CF id → PKJ id (length N_CF)
    cf2pkj_map = cp.full(N_CF, -1, dtype=cp.int32)
    cf2pkj_map[cfpkj.pre_idx] = cfpkj.post_idx

    # ----------------- main loop -----------------
    for step in tqdm(range(T_steps), desc="Simulating", unit="s"):
        sim_t = step * dt

        # IO: gap coupling + tonic bias
        I_gap = apply_ohmic_coupling(IO.V, pairs, cfg["g_gap_IO"])
        I_ext_io = cp.full(IO.V.size, cfg["io_bias_current"], dtype=cp.float32)
        IO = lif_step(IO, I_syn=I_gap, I_ext=I_ext_io, dt=dt, lif_cfg=cfg["lif_IO"])
        cf_spike = IO.spike.copy()
        if cf_spike.any():
            cfpkj.enqueue_from_pre_spikes(cf_spike)

        # PF & MF drives
        pf_spike_pool = step_pf_coinflip(pf_state, cfg["pf_rate_hz"], dt, rng)
        rec.log_spikes("PF", pf_spike_pool)
        if pf_spike_pool.any():
            pfpkj.enqueue_from_pre_spikes(pf_spike_pool, scale=pf_con_g)
            pfbc.enqueue_from_pre_spikes(pf_spike_pool)

        mf_spike = step_mf_poisson(N_DCN, cfg["mf_rate_hz"], dt, rng)
        if mf_spike.any():
            mfdcn.enqueue_from_pre_spikes(mf_spike)

        # Basket cells
        pfbc.step_decay()
        I_bc = simulate_current_from_proj(pfbc, BC.V)
        BC = lif_step(BC, I_syn=I_bc, I_ext=cp.zeros(N_BC, dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_BC"])
        if BC.spike.any():
            bcpkj.enqueue_from_pre_spikes(BC.spike)

        # PKJ: sum currents from CF, PF, BC
        cfpkj.step_decay(); bcpkj.step_decay(); pfpkj.step_decay()
        I_pkj = simulate_current_from_proj(cfpkj, PKJ.V)
        I_pkj += simulate_current_from_proj(bcpkj, PKJ.V)
        I_pkj += simulate_current_from_proj(pfpkj, PKJ.V)
        PKJ = lif_step(PKJ, I_syn=I_pkj, I_ext=cp.zeros(N_PKJ, dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_PKJ"])

        # DCN: inhibitory from PKJ + excitatory from MF
        if PKJ.spike.any():
            pkjdcn.enqueue_from_pre_spikes(PKJ.spike)
        pkjdcn.step_decay(); mfdcn.step_decay()
        I_dcn = simulate_current_from_proj(pkjdcn, DCN.V)
        I_dcn += simulate_current_from_proj(mfdcn, DCN.V)
        DCN = lif_step(DCN, I_syn=I_dcn, I_ext=cp.zeros(N_DCN, dtype=cp.float32), dt=dt, lif_cfg=cfg["lif_DCN"])

        # Plasticity: build CF→PKJ mask for this step
        cf_to_pkj_mask = cp.zeros(N_PKJ, dtype=bool)
        if cf_spike.any():
            cf_idx = cp.where(cf_spike)[0]
            pkj_targets = cf2pkj_map[cf_idx]
            cf_to_pkj_mask[pkj_targets] = True

        # STDP (last-spike) update on PF→PKJ
        pfpkj.w, traces = update_pfpkj_plasticity(
            pfpkj.w, pfpkj.pre_idx, pfpkj.post_idx,
            pf_spike_pool, PKJ.spike, cf_to_pkj_mask,
            dt, cfg, traces=traces
        )

        # Record
        rec.log_spikes("IO",  IO.spike)
        rec.log_spikes("PKJ", PKJ.spike)
        rec.log_spikes("BC",  BC.spike)
        rec.log_spikes("DCN", DCN.spike)
        rec.maybe_log_weights(step, pfpkj.w)

        # optional progress msg every sim-second
        if step % int(1.0 / dt) == 0 and step > 0:
            tqdm.write(f"Reached {sim_t:.1f} s simulated")

    # -------- save once at end --------
    out_npz = "cbm_py_output.npz"
    rec.finalize_npz(out_npz)
    return {"out_npz": os.path.abspath(out_npz)}


if __name__ == "__main__":
    info = run()
    print("Saved outputs to", info["out_npz"])