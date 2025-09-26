#!/bin/bash
"""
Clean up script for cerebellar microcircuit simulation.
Automatically deletes analysis outputs and simulation data files.
"""

echo "🧹 Cleaning up simulation outputs..."

# Delete analysis output directory and all contents
if [ -d "analysis_outputs" ]; then
    echo "  📁 Removing analysis_outputs/ directory..."
    rm -rf analysis_outputs/
    echo "  ✅ analysis_outputs/ removed"
else
    echo "  ℹ️  analysis_outputs/ directory not found"
fi

# Delete simulation output file
if [ -f "cbm_py_output.npz" ]; then
    echo "  📄 Removing cbm_py_output.npz..."
    rm -f cbm_py_output.npz
    echo "  ✅ cbm_py_output.npz removed"
else
    echo "  ℹ️  cbm_py_output.npz not found"
fi

# Delete any other common output files
if [ -f "*.png" ]; then
    echo "  🖼️  Removing PNG files..."
    rm -f *.png
    echo "  ✅ PNG files removed"
fi

if [ -f "*.npz" ]; then
    echo "  📦 Removing other NPZ files..."
    rm -f *.npz
    echo "  ✅ NPZ files removed"
fi

echo "🎉 Cleanup complete! All simulation outputs have been removed."
echo "   Ready for a fresh simulation run."
