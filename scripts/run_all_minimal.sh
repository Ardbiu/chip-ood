#!/bin/bash
# Minimal reproduction script

SEEDS=(0 1 2)
METHODS=(erm chip)

for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "Running $method seed $seed"
        bash scripts/run_colored_mnist.sh $method $seed
    done
done

echo "Aggregating results..."
.venv/bin/python -m chip_ood.evaluation.aggregate --input_dir results --output_dir results/aggregate
