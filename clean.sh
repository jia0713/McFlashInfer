#!/bin/bash

# Clean build directories
echo "Cleaning build directories..."
rm -rf build/
rm -rf csrc/generated/
rm -rf ~/.cache/flashinfer/
rm -rf flashinfer/data/
rm -rf dist/*
rm -rf aot-ops

echo "Clean complete!"