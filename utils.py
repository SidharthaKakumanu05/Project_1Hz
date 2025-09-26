#!/usr/bin/env python3
"""
Utility functions for the cerebellar microcircuit simulation.
Common functions used across multiple files.
"""

import cupy as cp
import numpy as np

def to_numpy(arr, dtype=None):
    """
    Convert input to a NumPy array, handling CuPy (GPU) arrays.
    
    Parameters
    ----------
    arr : array
        Input array (CuPy or NumPy)
    dtype : numpy dtype, optional
        Desired output dtype
        
    Returns
    -------
    numpy.ndarray
        NumPy array on CPU
    """
    if hasattr(arr, "get"):
        arr = arr.get()
    if dtype is not None:
        return np.asarray(arr, dtype=dtype)
    return np.asarray(arr)

def to_cupy(arr):
    """
    Convert input to a CuPy array, handling NumPy arrays.
    
    Parameters
    ----------
    arr : array
        Input array (NumPy or CuPy)
        
    Returns
    -------
    cupy.ndarray
        CuPy array on GPU
    """
    if hasattr(arr, "get"):
        return arr
    return cp.asarray(arr)
