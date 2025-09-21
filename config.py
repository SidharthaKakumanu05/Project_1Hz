import cupy as cp

def get_config():
    """
    Return a dictionary of all simulation parameters.
    This is the single source of truth for:
    - network sizes
    - time resolution and duration
    - neuron model constants
    - synapse properties
    - input firing rates
    - plasticity rules
    - recording settings
    """
    cfg = {}

    # -----------------------------
    # Network population sizes
    # -----------------------------
    cfg["N_BC"]  = 512     # basket cells
    cfg["N_PKJ"] = 128     # Purkinje cells
    cfg["N_CF"]  = 16      # climbing fibers
    cfg["N_DCN"] = 8       # deep cerebellar nuclei
    cfg["N_PF_POOL"] = 4096  # parallel fiber pool size

    # -----------------------------
    # Time settings
    # -----------------------------
    cfg["dt"] = 1e-4                  # time step (0.1 ms)
    cfg["T_sec"] = 5                # total simulation length in seconds
    cfg["T_steps"] = int(cfg["T_sec"] / cfg["dt"])  # number of simulation steps
    cfg["seed"] = 12345               # random seed (reproducibility)

    # -----------------------------
    # Leaky Integrate-and-Fire (LIF) neuron parameters
    # Units: capacitance (F), conductance (S), voltage (V)
    # -----------------------------
    cfg["lif_IO"] = dict(C=1.0e-9, gL=5.4e-9, EL=-0.067,
                         Vth=-0.052, Vreset=-0.070,
                         refrac_steps=int(0.002 / cfg["dt"]))  # 2 ms refractory

    cfg["lif_PKJ"] = dict(C=250e-12, gL=14e-9, EL=-0.068,
                          Vth=-0.052, Vreset=-0.072,
                          refrac_steps=int(0.002 / cfg["dt"]))

    cfg["lif_BC"]  = dict(C=180e-12, gL=10e-9, EL=-0.065,
                          Vth=-0.050, Vreset=-0.070,
                          refrac_steps=int(0.0015 / cfg["dt"]))

    cfg["lif_DCN"] = dict(C=220e-12, gL=10e-9, EL=-0.060,
                          Vth=-0.048, Vreset=-0.062,
                          refrac_steps=int(0.002 / cfg["dt"]))

    # -----------------------------
    # Synapse model parameters
    # tau = decay time constant (s)
    # E_rev = reversal potential (V)
    # delay_steps = synaptic delay (# of dt steps)
    # -----------------------------
    cfg["syn_PF_PKJ"] = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_PF_BC"]  = dict(tau=5e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_BC_PKJ"] = dict(tau=10e-3, E_rev=-0.075, delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_CF_PKJ"] = dict(tau=2e-3,  E_rev=0.0,    delay_steps=int(2e-3 / cfg["dt"]))
    cfg["syn_PKJ_DCN"]= dict(tau=12e-3, E_rev=-0.075, delay_steps=int(2e-3 / cfg["dt"]))
    cfg["syn_MF_DCN"] = dict(tau=6e-3,  E_rev=0.0,    delay_steps=int(1e-3 / cfg["dt"]))
    cfg["syn_DCN_IO"] = dict(tau=15e-3, E_rev=-0.075, delay_steps=int(3e-3 / cfg["dt"]))

    # -----------------------------
    # Input rates and conductance distributions
    # -----------------------------
    cfg["pf_rate_hz"] = 30.0                        # average PF firing rate
    cfg["pf_refrac_steps"] = int(0.005 / cfg["dt"]) # PF refractory (5 ms)
    cfg["pf_g_mean"] = 4e-10                        # mean PF conductance
    cfg["pf_g_std"]  = 2e-10                        # std dev of PF conductance

    cfg["mf_rate_hz"] = 50.0                        # average MF firing rate
    cfg["mf_g_mean"] = 4e-9                         # mean MF conductance
    cfg["mf_g_std"]  = 1e-9                         # std dev of MF conductance

    cfg["bc_g_mean"] = 5e-9                         # BC conductance mean
    cfg["bc_g_std"]  = 1e-9                         # BC conductance std dev
    cfg["bc_pkj_g_scale"] = 0.5                     # scaling for BC→PKJ inhibition

    # -----------------------------
    # PF→PKJ initial weights & limits
    # -----------------------------
    cfg["w_pfpkj_init"] = 1.0   # initial weight
    cfg["w_min"] = 0.2          # lower bound
    cfg["w_max"] = 3.0          # upper bound
    cfg["w_leak"] = 0.0         # unused here (reserved for weight decay)

    # -----------------------------
    # IO (Inferior Olive) excitability
    # -----------------------------
    cfg["io_bias_current"] = 83e-12       # constant bias current (pA scale)
    cfg["io_noise_std"] = 2e-12           # current noise (diffusion-like)
    cfg["io_bias_jitter_std"] = 0.3e-12   # static jitter per-neuron

    # -----------------------------
    # DCN (Deep Cerebellar Nuclei) excitability
    # -----------------------------
    # DCN rheobase: gL * (Vth - EL) = 10e-9 * (-0.048 - (-0.060)) = 1.2e-10 A
    # Add moderate bias current to keep DCN neurons near threshold
    cfg["dcn_bias_current"] = 0.8e-10     # bias current (80 pA, below rheobase)

    # -----------------------------
    # Gap-junction coupling between IO neurons
    # -----------------------------
    cfg["g_gap_IO"] = 1e-10   # not too strong, avoids full synchronization

    # -----------------------------
    # Plasticity parameters
    # -----------------------------
    cfg["ltd_window"] = dict(t_pre_cf=0.05)  # LTD window = 50 ms
    cfg["ltp_window"] = dict(t_pre_cf=0.45)  # LTP window = 450 ms
    cfg["ltd_scale"]  = 9e-4                 # LTD decrement size
    cfg["ltp_scale"]  = 1e-4                 # LTP increment size

    # -----------------------------
    # Recording frequency
    # -----------------------------
    cfg["rec_weight_every_steps"] = int(0.01 / cfg["dt"])  # record weights every 10 ms

    return cfg