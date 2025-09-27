import cupy as cp
import numpy as np

def compute_coupling_currents(V, pairs, coupling_strength):
    I = cp.zeros_like(V, dtype=cp.float32)
    
    for i, j in pairs:
        dV = V[j] - V[i]
        I[i] += coupling_strength * dV
        I[j] -= coupling_strength * dV
    
    return I

# Pre-computed coupling connections for better performance
_io_coupling_pairs = None

def compute_io_coupling_currents(V, coupling_strength, num_neighbors=3):
    global _io_coupling_pairs, _io_coupling_indices, _io_coupling_weights
    N = V.shape[0]
    I = cp.zeros_like(V, dtype=cp.float32)
    
    if N <= 1:
        return I
    
    # Pre-compute coupling pairs and weights only once
    if _io_coupling_pairs is None or len(_io_coupling_pairs) != N:
        np.random.seed(12345)
        _io_coupling_pairs = []
        _io_coupling_indices = []
        _io_coupling_weights = []
        
        for i in range(N):
            neighbors = []
            available = list(range(N))
            available.remove(i)
            
            for _ in range(min(num_neighbors, len(available))):
                j = np.random.choice(available)
                neighbors.append(j)
                available.remove(j)
            _io_coupling_pairs.append(neighbors)
            
            # Pre-compute indices and weights for vectorized operations
            for j in neighbors:
                _io_coupling_indices.extend([(i, j), (j, i)])
                _io_coupling_weights.extend([coupling_strength, -coupling_strength])
        
        # Convert to CuPy arrays for GPU operations
        _io_coupling_indices = cp.array(_io_coupling_indices, dtype=cp.int32)
        _io_coupling_weights = cp.array(_io_coupling_weights, dtype=cp.float32)
    
    # Ultra-optimized vectorized coupling calculation using add.at
    if len(_io_coupling_indices) > 0:
        i_indices = _io_coupling_indices[:, 0]
        j_indices = _io_coupling_indices[:, 1]
        dV = V[j_indices] - V[i_indices]
        I_contrib = _io_coupling_weights * dV
        # Use add.at for maximum performance with sparse updates
        cp.add.at(I, i_indices, I_contrib)
    
    return I