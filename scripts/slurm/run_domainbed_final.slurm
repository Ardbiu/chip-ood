#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p engaging
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o logs/final_%j.out
#SBATCH -e logs/final_%j.err

# Load Modules (Specific to Engaging Cluster)
module load deprecated-modules
module load gcc/12.2.0-x86_64
module load python/3.10.8-x86_64

source .venv/bin/activate

# Example: Run 10 seeds for best CHIP config
# User should edit this or pass args
DATASET=$1
METHOD=$2
ARGS=$3

echo "Running Final: $METHOD on $DATASET"
for seed in {0..9}; do
    python -m chip_ood.training.trainer \
        data=$DATASET method=$METHOD seed=$seed \
        output_dir="results/domainbed_final/${DATASET}_${METHOD}_${seed}" \
        $ARGS
done
