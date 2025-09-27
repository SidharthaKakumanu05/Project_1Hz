import cupy as cp
import numpy as np

def compute_coupling_currents(V, pairs, coupling_strength):
    I = cp.zeros_like(V, dtype=cp.float32)
    
    for i, j in pairs:
        dV = V[j] - V[i]
        I[i] += coupling_strength * dV
        I[j] -= coupling_strength * dV
    
    return I

def compute_io_coupling_currents(V, coupling_strength, num_neighbors=3):
    N = V.shape[0]
    I = cp.zeros_like(V, dtype=cp.float32)
    
    if N <= 1:
        return I
    
    np.random.seed(12345)
    for i in range(N):
        neighbors = []
        available = list(range(N))
        available.remove(i)
        
        for _ in range(min(num_neighbors, len(available))):
            j = np.random.choice(available)
            neighbors.append(j)
            available.remove(j)
        
        for j in neighbors:
            dV = V[j] - V[i]
            I[i] += coupling_strength * dV
            I[j] -= coupling_strength * dV
    
    return I