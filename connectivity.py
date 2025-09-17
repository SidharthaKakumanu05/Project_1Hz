import cupy as cp


def build_connectivity(cfg, rng):
    """
    Build connectivity matrices for cerebellar microcircuit with
    convergence ratios scaled from the big simulation.
    """

    graph = {}

    N_BC, N_PKJ, N_CF, N_DCN, N_PF = (
        cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"], cfg["N_PF_POOL"]
    )

    # --- CF → PKJ (1:1) ---
    cf_pre = cp.arange(N_CF, dtype=cp.int32)
    pkj_post = cp.arange(N_CF, dtype=cp.int32)  # assumes N_CF <= N_PKJ
    graph["CF_to_PKJ"] = {"pre_idx": cf_pre, "post_idx": pkj_post}

    # --- PF → PKJ ---
    # Big sim: ~100k PF → 1 PKJ. Scaled: all PFs (4096) connect to each PKJ.
    pf_pre = cp.repeat(cp.arange(N_PF, dtype=cp.int32), N_PKJ)
    pkj_post = cp.tile(cp.arange(N_PKJ, dtype=cp.int32), N_PF)
    graph["PF_to_PKJ"] = {"pre_idx": pf_pre, "post_idx": pkj_post}

    # --- PF → BC ---
    # Similar density: all PFs connect to all BCs.
    pf_pre = cp.repeat(cp.arange(N_PF, dtype=cp.int32), N_BC)
    bc_post = cp.tile(cp.arange(N_BC, dtype=cp.int32), N_PF)
    graph["PF_to_BC"] = {"pre_idx": pf_pre, "post_idx": bc_post}

    # --- BC → PKJ ---
    # Each PKJ receives ~20 BC inputs. Each BC projects to ~5 PKJs.
    n_bcpkj = N_PKJ * 20
    bc_pre = rng.integers(0, N_BC, size=n_bcpkj, dtype=cp.int32)
    pkj_post = rng.integers(0, N_PKJ, size=n_bcpkj, dtype=cp.int32)
    graph["BC_to_PKJ"] = {"pre_idx": bc_pre, "post_idx": pkj_post}

    # --- PKJ → DCN ---
    # Each DCN receives inhibitory input from many PKJs (scaled down here).
    n_pkjdcn = N_DCN * (N_PKJ // 2)  # each DCN gets ~half of PKJs
    pkj_pre = rng.integers(0, N_PKJ, size=n_pkjdcn, dtype=cp.int32)
    dcn_post = cp.repeat(cp.arange(N_DCN, dtype=cp.int32), n_pkjdcn // N_DCN)
    graph["PKJ_to_DCN"] = {"pre_idx": pkj_pre, "post_idx": dcn_post}

    # --- MF → DCN ---
    # Each DCN gets ~50 mossy fiber inputs (scaled).
    n_mfdcn = N_DCN * 50
    mf_pre = rng.integers(0, N_DCN, size=n_mfdcn, dtype=cp.int32)  # fake MFs
    dcn_post = cp.repeat(cp.arange(N_DCN, dtype=cp.int32), 50)
    graph["MF_to_DCN"] = {"pre_idx": mf_pre, "post_idx": dcn_post}

    return graph