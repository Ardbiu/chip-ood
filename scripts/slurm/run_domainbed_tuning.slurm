#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -p mit_normal_gpu
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --array=1-6%20 
#SBATCH -o logs/tuning_%A_%a.out
#SBATCH -e logs/tuning_%A_%a.err

# Usage: sbatch scripts/slurm/run_domainbed_tuning.slurm
# Expects: scripts/generated_tuning_cmds.txt to exist

# Load Modules (Specific to Engaging Cluster)
module load deprecated-modules
module load gcc/12.2.0-x86_64
module load python/3.10.8-x86_64

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Get command from line number
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/generated_tuning_cmds.txt)
echo "Running: $CMD"
eval $CMD
