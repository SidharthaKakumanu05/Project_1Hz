
import numpy as np

def build_connectivity(cfg, rng):
    N_BC, N_PKJ, N_CF, N_DCN = cfg["N_BC"], cfg["N_PKJ"], cfg["N_CF"], cfg["N_DCN"]
    N_PF_POOL = cfg["N_PF_POOL"]

    graph = {}

    pkj_idx = rng.choice(N_PKJ, size=N_CF, replace=False)
    graph["CF_to_PKJ"] = dict(pre_idx=np.arange(N_CF, dtype=np.int32),
                              post_idx=pkj_idx.astype(np.int32))

    pkj_targets = []
    dcn_targets = []
    assign = np.arange(N_PKJ); rng.shuffle(assign)
    chunks = np.array_split(assign, N_DCN)
    for dcn, chunk in enumerate(chunks):
        for p in chunk:
            pkj_targets.append(p); dcn_targets.append(dcn)
    graph["PKJ_to_DCN"] = dict(pre_idx=np.array(pkj_targets, dtype=np.int32),
                               post_idx=np.array(dcn_targets, dtype=np.int32))

    K_pf_pkj = 256
    pf_pre = []; pf_post = []
    for p in range(N_PKJ):
        chosen = rng.integers(0, N_PF_POOL, size=K_pf_pkj)
        pf_pre.append(chosen)
        pf_post.append(np.full(K_pf_pkj, p, dtype=np.int32))
    graph["PF_to_PKJ"] = dict(pre_idx=np.concatenate(pf_pre).astype(np.int32),
                              post_idx=np.concatenate(pf_post).astype(np.int32))

    K_pf_bc = 128
    pf_pre = []; bc_post = []
    for b in range(N_BC):
        chosen = rng.integers(0, N_PF_POOL, size=K_pf_bc)
        pf_pre.append(chosen)
        bc_post.append(np.full(K_pf_bc, b, dtype=np.int32))
    graph["PF_to_BC"] = dict(pre_idx=np.concatenate(pf_pre).astype(np.int32),
                             post_idx=np.concatenate(bc_post).astype(np.int32))

    bc_pre = []; pkj_post = []
    for p in range(N_PKJ):
        chosen = rng.choice(N_BC, size=4, replace=False)
        bc_pre.append(chosen)
        pkj_post.append(np.full(4, p, dtype=np.int32))
    graph["BC_to_PKJ"] = dict(pre_idx=np.concatenate(bc_pre).astype(np.int32),
                              post_idx=np.concatenate(pkj_post).astype(np.int32))

    graph["MF_to_DCN"] = dict(pre_idx=np.arange(N_DCN, dtype=np.int32),
                              post_idx=np.arange(N_DCN, dtype=np.int32))
    return graph
