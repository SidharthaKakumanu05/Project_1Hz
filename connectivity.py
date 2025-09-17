import cupy as cp


def build_connectivity(cfg):
    """
    Build connectivity matrices with fixed convergence ratios.
    """
    N_BC  = cfg["N_BC"]
    N_PKJ = cfg["N_PKJ"]
    N_CF  = cfg["N_CF"]
    N_DCN = cfg["N_DCN"]
    N_PF  = cfg["N_PF_POOL"]

    graph = {}

    # CF → PKJ (1:1)
    cf_pre = cp.arange(N_CF, dtype=cp.int32)
    cf_post = cp.arange(N_CF, dtype=cp.int32) % N_PKJ
    graph["CF_to_PKJ"] = {"pre_idx": cf_pre, "post_idx": cf_post}

    # PF → PKJ (each PKJ gets many PFs)
    fan_in_pfpkj = N_PF // N_PKJ
    pf_pre = cp.random.randint(0, N_PF, size=fan_in_pfpkj * N_PKJ, dtype=cp.int32)
    pf_post = cp.repeat(cp.arange(N_PKJ, dtype=cp.int32), fan_in_pfpkj)
    graph["PF_to_PKJ"] = {"pre_idx": pf_pre, "post_idx": pf_post}

    # PF → BC (each BC gets many PFs)
    fan_in_pfbc = N_PF // N_BC
    pfbc_pre = cp.random.randint(0, N_PF, size=fan_in_pfbc * N_BC, dtype=cp.int32)
    pfbc_post = cp.repeat(cp.arange(N_BC, dtype=cp.int32), fan_in_pfbc)
    graph["PF_to_BC"] = {"pre_idx": pfbc_pre, "post_idx": pfbc_post}

    # BC → PKJ (inhibitory, random sparse)
    bc_conn = N_BC // 4
    bc_pre = cp.random.randint(0, N_BC, size=bc_conn * N_PKJ, dtype=cp.int32)
    bc_post = cp.repeat(cp.arange(N_PKJ, dtype=cp.int32), bc_conn)
    graph["BC_to_PKJ"] = {"pre_idx": bc_pre, "post_idx": bc_post}

    # PKJ → DCN (each DCN gets input from many PKJs)
    pkj_conn = N_PKJ // N_DCN
    pkj_pre = cp.random.randint(0, N_PKJ, size=pkj_conn * N_DCN, dtype=cp.int32)
    pkj_post = cp.repeat(cp.arange(N_DCN, dtype=cp.int32), pkj_conn)
    graph["PKJ_to_DCN"] = {"pre_idx": pkj_pre, "post_idx": pkj_post}

    # MF → DCN (each DCN gets MF inputs)
    mf_pre = cp.arange(N_DCN, dtype=cp.int32)
    mf_post = cp.arange(N_DCN, dtype=cp.int32)
    graph["MF_to_DCN"] = {"pre_idx": mf_pre, "post_idx": mf_post}

    return graph