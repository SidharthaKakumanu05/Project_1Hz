import cupy as cp

def get_config():
    cfg = {}
    cfg["N_BC"]  = 512
    cfg["N_PKJ"] = 128
    cfg["N_CF"]  = 16
    cfg["N_DCN"] = 8
    cfg["N_PF_POOL"] = 4096

    cfg["dt"] = 1e-4
    cfg["T_sec"] = 300
    cfg["T_steps"] = int(cfg["T_sec"] / cfg["dt"])
    cfg["seed"] = 12345

    # LIF neuron configs
    cfg["lif_IO"] = dict(C=1.0e-9, gL=5.4e-9,EL=-0.067,
                     Vth=-0.052, Vreset=-0.070,
                     refrac_steps=int(0.002 / cfg["dt"]))
    cfg["lif_PKJ"] = dict(C=250e-12, gL=14e-9, EL=-0.068,
                          Vth=-0.052, Vreset=-0.072,
                          refrac_steps=int(0.002 / cfg["dt"]))
    cfg["lif_BC"]  = dict(C=180e-12, gL=10e-9, EL=-0.065,
                          Vth=-0.050, Vreset=-0.070,
                          refrac_steps=int(0.0015 / cfg["dt"]))
    cfg["lif_DCN"] = dict(C=220e-12, gL=10e-9, EL=-0.060,
                          Vth=-0.048, Vreset=-0.062,
                          refrac_steps=int(0.002 / cfg["dt"]))

    # Synapse configs
    cfg["syn_PF_PKJ"] = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_PF_BC"]  = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_BC_PKJ"] = dict(tau=10e-3, E_rev=-0.075, delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_CF_PKJ"] = dict(tau=2e-3,  E_rev=0.0,    delay_steps=int(2e-3 / cfg["dt"]))
    cfg["syn_PKJ_DCN"]= dict(tau=12e-3, E_rev=-0.075, delay_steps=int(2e-3 / cfg["dt"]))
    cfg["syn_MF_DCN"] = dict(tau=6e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))

    # Input rates & conductances
    cfg["pf_rate_hz"] = 30.0
    cfg["pf_refrac_steps"] = int(0.005 / cfg["dt"])
    cfg["pf_g_mean"] = 4e-10
    cfg["pf_g_std"]  = 2e-10
    cfg["mf_rate_hz"] = 50.0
    cfg["mf_g_mean"] = 4e-9
    cfg["mf_g_std"]  = 1e-9
    cfg["bc_g_mean"] = 5e-9    # adjust as needed
    cfg["bc_g_std"]  = 1e-9

    # Initial PF→PKJ weights
    cfg["w_pfpkj_init"] = 1.0
    cfg["w_min"] = 0.2
    cfg["w_max"] = 3.0
    cfg["w_leak"] = 0.0

    # --- IO excitability (near threshold) ---
    cfg["io_bias_current"] = 82e-12     # ~subthreshold; you were at 181e-12
    cfg["io_noise_std"] = 2e-12      # A/√s, diffusion-style current noise
    cfg["io_bias_jitter_std"] = 0.3e-12      # A, fixed per-neuron offset

    # Gap coupling (not too strong or they'll synchronize too much)
    cfg["g_gap_IO"] = 1e-10


    # Plasticity windows & scales
    cfg["ltd_window"] = dict(t_pre_cf=0.05)   # 50 ms LTD
    cfg["ltp_window"] = dict(t_pre_cf=0.45)   # 450 ms LTP
    cfg["ltd_scale"]  = 9e-4                 # LTD strong
    cfg["ltp_scale"]  = 1e-4                 # LTP weak

    # Recording frequency
    cfg["rec_weight_every_steps"] = int(0.01 / cfg["dt"])

    return cfg