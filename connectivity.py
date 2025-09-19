import cupy as cp
import numpy as np

def build_connectivity(cfg):
    N_BC, N_PKJ, N_CF, N_DCN = cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"]
    rng = np.random.default_rng(cfg["seed"])
    graph = {}

    # PF → PKJ (random convergent)
    pf_conn = cfg["N_PF_POOL"] // 4
    pf_pre = rng.integers(0, cfg["N_PF_POOL"], size=pf_conn * N_PKJ, dtype=np.int32)
    pf_post = np.repeat(np.arange(N_PKJ, dtype=np.int32), pf_conn)
    graph["PF_to_PKJ"] = {
        "pre_idx": cp.asarray(pf_pre),
        "post_idx": cp.asarray(pf_post),
    }

    # PF → BC
    pfbc_conn = cfg["N_PF_POOL"] // 8
    pfbc_pre = rng.integers(0, cfg["N_PF_POOL"], size=pfbc_conn * N_BC, dtype=np.int32)
    pfbc_post = np.repeat(np.arange(N_BC, dtype=np.int32), pfbc_conn)
    graph["PF_to_BC"] = {
        "pre_idx": cp.asarray(pfbc_pre),
        "post_idx": cp.asarray(pfbc_post),
    }

    # CF → PKJ (1-to-1 spread across PKJs)
    cf_pre = np.arange(N_CF, dtype=np.int32)
    cf_post = np.linspace(0, N_PKJ - 1, N_CF, dtype=np.int32)
    graph["CF_to_PKJ"] = {
        "pre_idx": cp.asarray(cf_pre),
        "post_idx": cp.asarray(cf_post),
    }

    # BC → PKJ (sparse with inhibitory conductances)
    bc_conn = N_BC // 4
    bc_pre  = rng.integers(0, N_BC, size=bc_conn * N_PKJ, dtype=np.int32)
    bc_post = np.repeat(np.arange(N_PKJ, dtype=np.int32), bc_conn)
    bc_g = rng.normal(cfg["bc_g_mean"], cfg["bc_g_std"], size=bc_conn * N_PKJ)
    bc_g = bc_g * cfg["bc_pkj_g_scale"]
    bc_g = cp.asarray(np.clip(bc_g, 0, None), dtype=cp.float32)

    graph["BC_to_PKJ"] = {
        "pre_idx": cp.asarray(bc_pre),
        "post_idx": cp.asarray(bc_post),
        "g": bc_g,
    }

    # PKJ → DCN (random sparse)
    pkj_conn = N_PKJ // 8
    pkj_pre  = rng.integers(0, N_PKJ, size=pkj_conn * N_DCN, dtype=np.int32)
    pkj_post = np.repeat(np.arange(N_DCN, dtype=np.int32), pkj_conn)
    graph["PKJ_to_DCN"] = {
        "pre_idx": cp.asarray(pkj_pre),
        "post_idx": cp.asarray(pkj_post),
    }

    # MF → DCN (make presyn size = N_DCN so spikes are length N_DCN)
    # simple 1:1 for now (can fan-in later if needed)
    graph["MF_to_DCN"] = {
        "pre_idx": cp.arange(N_DCN, dtype=cp.int32),
        "post_idx": cp.arange(N_DCN, dtype=cp.int32),
    }

    return graph
