#!/bin/bash
# fast sanity check

METHOD=${1:-erm}
SEED=${2:-0}

echo "Running Colored MNIST with $METHOD, Seed $SEED"

.venv/bin/python -m chip_ood.training.trainer \
    method=$METHOD \
    data=colored_mnist \
    trainer.max_epochs=5 \
    seed=$SEED \
    output_dir=results
