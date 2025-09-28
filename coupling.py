import cupy as cp
import numpy as np

def compute_coupling_currents(V, pairs, coupling_strength):
    I = cp.zeros_like(V, dtype=cp.float32)
    
    for i, j in pairs:
        dV = V[j] - V[i]
        I[i] += coupling_strength * dV
        I[j] -= coupling_strength * dV
    
    return I

class IOCouplingOptimized:
    """
    Optimized IO coupling computation that pre-computes neighbor connections
    and uses vectorized operations for maximum performance.
    """
    def __init__(self, N, coupling_strength, num_neighbors=3, seed=12345):
        self.N = N
        self.coupling_strength = coupling_strength
        self.num_neighbors = num_neighbors
        
        if N <= 1:
            self.pre_idx = cp.array([], dtype=cp.int32)
            self.post_idx = cp.array([], dtype=cp.int32)
            return
            
        # Pre-compute neighbor connections using numpy for reproducibility
        np.random.seed(seed)
        pre_indices = []
        post_indices = []
        
        for i in range(N):
            available = list(range(N))
            available.remove(i)
            neighbors = np.random.choice(available, size=min(num_neighbors, len(available)), replace=False)
            
            for j in neighbors:
                pre_indices.append(i)
                post_indices.append(j)
        
        # Convert to CuPy arrays for GPU computation
        self.pre_idx = cp.array(pre_indices, dtype=cp.int32)
        self.post_idx = cp.array(post_indices, dtype=cp.int32)
    
    def compute_coupling_currents(self, V):
        """
        Compute coupling currents using pre-computed connections and vectorized operations.
        This is much faster than the original implementation.
        """
        if self.N <= 1:
            return cp.zeros_like(V, dtype=cp.float32)
        
        # Vectorized computation: dV = V[post] - V[pre]
        dV = V[self.post_idx] - V[self.pre_idx]
        
        # Initialize current array
        I = cp.zeros_like(V, dtype=cp.float32)
        
        # Vectorized current computation using scatter operations
        cp.add.at(I, self.pre_idx, self.coupling_strength * dV)
        cp.add.at(I, self.post_idx, -self.coupling_strength * dV)
        
        return I

# Global coupling object for reuse
_io_coupling = None

def compute_io_coupling_currents(V, coupling_strength, num_neighbors=3):
    """
    Optimized IO coupling computation that reuses pre-computed connections.
    This function maintains the same interface as the original but is much faster.
    """
    global _io_coupling
    
    N = V.shape[0]
    
    # Initialize or update coupling object if needed
    if _io_coupling is None or _io_coupling.N != N or _io_coupling.coupling_strength != coupling_strength:
        _io_coupling = IOCouplingOptimized(N, coupling_strength, num_neighbors)
    
    return _io_coupling.compute_coupling_currents(V)