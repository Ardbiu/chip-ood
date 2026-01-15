# CHIP (Causal–spurious Hidden Information Partitioning) - ICML OOD Project

This repository contains the official implementation of **CHIP**, a method for out-of-distribution (OOD) generalization via split representations ($Z_c, Z_s$).

## Installation

```bash
# Clone the repo
git clone <this-repo-url>
cd chip_ood

# Create environment (optional but recommended)
conda create -n chip python=3.10
conda activate chip

# Install dependencies
pip install -e .
```

## Quick Start: Colored MNIST

To run a fast sanity check on Colored MNIST (Train on correlation 0.9, Test on correlation 0.1):

```bash
# Run ERM baseline
bash scripts/run_colored_mnist.sh erm

# Run CHIP method
bash scripts/run_colored_mnist.sh chip
```

Results will be located in `results/colored_mnist_<method>_<seed>`.

## Reproducing Main Results

To reproduce the main tables from the paper:

1. **Download Datasets**:
   ```bash
   bash scripts/download_domainbed.sh
   # Sets up PACS, VLCS, etc.
   ```

2. **Run Experiments**:
   ```bash
   bash scripts/run_all_minimal.sh
   ```
   This script runs 5 seeds for ERM, CHIP, and EIIL on the configured datasets.

3. **Aggregate Results**:
   ```bash
   python -m chip_ood.evaluation.aggregate --input_dir results --output_dir results/aggregate
   ```
   This generates `results/aggregate/main_table.tex` and `.csv`.

## Structure

- `src/chip_ood/configs`: Hydra configs for methods, data, and training.
- `src/chip_ood/models`: Encoder, Decoder, and Adversary architectures.
- `src/chip_ood/methods`: Training logic for ERM, CHIP, EIIL.
- `src/chip_ood/data`: Data loaders and ColoredMNIST generator.

## Outputs

Every run produces:
- `metrics.jsonl`: Per-step logging.
- `final_metrics.json`: Summary stats.
- `config.yaml`: Reproduced config.
- `checkpoints/`: Model weights.

