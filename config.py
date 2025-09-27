import cupy as cp
import numpy as np
from numpy import exp

def get_config():
    cfg = {}
    
    # Population sizes matching CbmSim exactly
    cfg["N_BC"] = 512      # num_bc from CbmSim
    cfg["N_PKJ"] = 128      # num_pc from CbmSim (Purkinje cells)
    cfg["N_CF"] = 16       # Keep same as current
    cfg["N_DCN"] = 8       # num_nc from CbmSim
    cfg["N_PF_POOL"] = 4096  # Keep same as current
    
    cfg["dt"] = 1e-3       # msPerTimeStep = 1.0 ms in CbmSim
    cfg["T_sec"] = 600
    cfg["T_steps"] = int(cfg["T_sec"] / cfg["dt"])
    cfg["seed"] = 12345
    
    # IO parameters matching CbmSim exactly
    # CbmSim: rawGLeakIO=0.15, eLeakIO=-60.0, threshRestIO=-57.4, threshMaxIO=10.0
    cfg["io_params"] = dict(
        gL=0.03, EL=-60.0, Vth=-57.4, Vreset=-70.0,  # Scaled gL like CbmSim
        refrac_steps=int(0.002 / cfg["dt"]),
        thresh_decay=1-exp(-cfg["dt"]/200.0), thresh_rest=-57.4,  # threshDecTauIO=200.0
        thresh_max=10.0, eNCtoIO=-80.0
    )
    
    # PKJ parameters matching CbmSim exactly  
    # CbmSim: rawGLeakPC=0.2, eLeakPC=-60.0, threshRestPC=-60.62, threshMaxPC=-48.0
    cfg["pkj_params"] = dict(
        gL=0.2/(6-cfg["dt"]), EL=-60.0, Vth=-60.62, Vreset=-72.0,  # Scaled gL like CbmSim
        refrac_steps=int(0.002 / cfg["dt"]),
        thresh_decay=1-exp(-cfg["dt"]/6.0), thresh_rest=-60.62,  # threshDecTauPC=6.0
        thresh_max=-48.0, eBCtoPC=-80.0, eSCtoPC=-80.0  # threshMaxPC=-48.0
    )
    
    # BC parameters matching CbmSim exactly
    # CbmSim: rawGLeakBC=0.13, eLeakBC=-70.0, threshRestBC=-65.0, threshMaxBC=0.0
    cfg["bc_params"] = dict(
        gL=0.13, EL=-70.0, Vth=-65.0, Vreset=-70.0,  # rawGLeakBC not scaled in CbmSim
        refrac_steps=int(0.0015 / cfg["dt"]),
        thresh_decay=1-exp(-cfg["dt"]/10.0), thresh_rest=-65.0,  # threshDecTauBC=10.0
        thresh_max=0.0, ePCtoBC=-70.0
    )
    
    # DCN parameters matching CbmSim exactly
    # CbmSim: rawGLeakNC=0.1, eLeakNC=-65.0, threshRestNC=-72.0, threshMaxNC=-40.0
    cfg["dcn_params"] = dict(
        gL=0.1/(6-cfg["dt"]), EL=-65.0, Vth=-72.0, Vreset=-62.0,  # Scaled gL like CbmSim
        refrac_steps=int(0.01 / cfg["dt"]),
        thresh_decay=1-exp(-cfg["dt"]/5.0), thresh_rest=-72.0,  # threshDecTauNC=5.0
        thresh_max=-40.0, ePCtoNC=-80.0
    )
    
    # Synaptic parameters matching CbmSim exactly
    # CbmSim tau values: gDecTauGRtoPC=4.15, gDecTauGRtoBC=2.0, gDecTauBCtoPC=7.0, 
    # gDecTauSCtoPC=4.15, gDecTauPCtoNC=4.15, gDecTauMFtoNC=6.0, gIncTauNCtoIO=300.0
    cfg["synapses"] = {
        "PF_PKJ": dict(tau=4.15e-3, E_rev=0.0, delay_steps=int(1e-3 / cfg["dt"])),  # gDecTauGRtoPC
        "PF_BC": dict(tau=2.0e-3, E_rev=0.0, delay_steps=int(1e-3 / cfg["dt"])),   # gDecTauGRtoBC  
        "BC_PKJ": dict(tau=7.0e-3, E_rev=-80.0, delay_steps=int(1e-3 / cfg["dt"])), # gDecTauBCtoPC
        "CF_PKJ": dict(tau=2e-3, E_rev=0.0, delay_steps=int(2e-3 / cfg["dt"])),    # Keep current
        "PKJ_DCN": dict(tau=4.15e-3, E_rev=-80.0, delay_steps=int(2e-3 / cfg["dt"])), # gDecTauPCtoNC
        "MF_DCN": dict(tau=6.0e-3, E_rev=0.0, delay_steps=int(1e-3 / cfg["dt"])),   # gDecTauMFtoNC
        "DCN_IO": dict(tau=300.0e-3, E_rev=-80.0, delay_steps=int(3e-3 / cfg["dt"])) # gIncTauNCtoIO
    }
    
    # Input rates and conductances matching CbmSim
    cfg["pf_rate"] = 30.0
    cfg["pf_refrac_steps"] = int(0.005 / cfg["dt"])
    # CbmSim: gIncGRtoPC = 0.55e-05 (very small conductance)
    cfg["pf_g_mean"] = 0.55e-05  # gIncGRtoPC from CbmSim
    cfg["pf_g_std"] = 0.55e-05 * 0.1  # 10% std dev
    
    cfg["mf_rate"] = 10.0
    # CbmSim: rawGMFAMPAIncNC = 2.35, rawGMFNMDAIncNC = 2.35
    cfg["mf_g_mean"] = 2.35  # rawGMFAMPAIncNC from CbmSim
    cfg["mf_g_std"] = 0.1
    
    # CbmSim: gIncBCtoPC = 0.0003
    cfg["bc_g_mean"] = 0.0003  # gIncBCtoPC from CbmSim
    cfg["bc_g_std"] = 0.00003  # 10% std dev
    
    # CbmSim: initSynWofGRtoPC = 0.5
    cfg["weight_init"] = 1  # initSynWofGRtoPC from CbmSim
    cfg["weight_min"] = 0.2
    cfg["weight_max"] = 3.0
    
    # CbmSim: coupleRiRjRatioIO = 0.05
    cfg["io_coupling_strength"] = 0.025 # coupleRiRjRatioIO from CbmSim
    
    # Plasticity parameters matching CbmSim exactly
    cfg["ltd_window"] = 0.5
    cfg["ltp_window"] = 0.5
    # CbmSim: synLTDStepSizeGRtoPC = -0.00275, synLTPStepSizeGRtoPC = 0.00030556
    cfg["ltd_scale"] = -0.00275  # synLTDStepSizeGRtoPC
    cfg["ltp_scale"] = 0.00030556  # synLTPStepSizeGRtoPC
    
    cfg["rec_weight_every_steps"] = int(0.1 / cfg["dt"])
    cfg["plasticity_every_steps"] = int(0.01 / cfg["dt"])

    return cfg