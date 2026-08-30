#!/bin/bash
# Test and run Reservoir Computing

set -e

echo "========================================"
echo "Reservoir Computing Test Suite"
echo "========================================"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install numpy scikit-learn matplotlib

# Run tests
echo ""
echo "Running Reservoir Computing tests..."
python reservoir_computing.py

echo ""
echo "========================================"
echo "Tests completed successfully!"
echo "========================================"
