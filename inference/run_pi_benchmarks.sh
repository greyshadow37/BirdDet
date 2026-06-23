#!/bin/bash

# Setup colors
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting Bird Detection Benchmarks on Raspberry Pi ===${NC}"

# Navigate to the project root directory where the script resides
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$CDIR/.."

# Activate the Python 3.9 virtual environment
if [ -d "bird_venv" ]; then
    echo "Activating virtual environment: bird_venv"
    source bird_venv/bin/activate
elif [ -d "../bird_venv" ]; then
    echo "Activating virtual environment: ../bird_venv"
    source ../bird_venv/bin/activate
else
    echo "Warning: bird_venv not found. Running with current environment."
fi

# Run the inference script
echo -e "${GREEN}Running benchmarks on ONNX, TFLite, and NCNN models...${NC}"
python inference/run_inference.py \
    --weights_dir best_weights \
    --image pi_images \
    --benchmark \
    --num_runs 100 \
    --csv benchmark_results.csv

echo -e "${GREEN}=== Benchmarks completed successfully! ===${NC}"
echo "Results saved to: benchmark_results.csv"
