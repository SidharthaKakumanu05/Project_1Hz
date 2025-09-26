#!/bin/bash

# Cerebellar Microcircuit Simulation Runner
# This script runs the complete simulation pipeline:
# 1. Run the simulation (main.py)
# 2. Analyze the results (analysis.py)

echo "=========================================="
echo "Starting Cerebellar Microcircuit Simulation"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Run the simulation
echo "Step 1: Running simulation..."
echo "Command: python3 main.py"
python3 main.py

# Check if simulation was successful
if [ $? -eq 0 ]; then
    echo "✓ Simulation completed successfully"
else
    echo "✗ Simulation failed with exit code $?"
    exit 1
fi

echo ""
echo "Step 2: Running analysis..."
echo "Command: python3 analysis.py"
python3 analysis.py

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo "✓ Analysis completed successfully"
    echo ""
    echo "=========================================="
    echo "Simulation and Analysis Complete!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - Simulation data: cbm_py_output.npz"
    echo "  - Analysis plots: analysis_outputs/"
    echo ""
    echo "Generated plots:"
    echo "  - io_raster.png (IO neuron firing patterns)"
    echo "  - pkj_raster.png (Purkinje cell activity)"
    echo "  - dcn_raster.png (Deep cerebellar nuclei output)"
    echo "  - pkj_mean_rate.png (PKJ firing rate over time)"
    echo "  - weights_*.png (Synaptic weight evolution)"
    echo "=========================================="
else
    echo "✗ Analysis failed with exit code $?"
    exit 1
fi
