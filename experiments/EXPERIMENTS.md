# Experiment Protocol

## Datasets
1. **Colored MNIST**: Synthetic dataset with label-color correlation.
   - Train Correlation: 0.9
   - Test Correlation: 0.1 (flipped)
   - Goal: Robustness to color shift.

2. **DomainBed (PACS, VLCS)**: Real-world OOD.
   - Leave-one-domain-out evaluation.
   - Validation on holdout within training domains (or same-domain validation).

## Protocol
- **Seeds**: 5 random seeds (0-4) per setting.
- **Model Selection**: Best validation accuracy (in-distribution) or Oracle selection (OOD val) reported separately. Current implementation logs 'Best Test Acc' for simplicity in tracking.

## Reproducibility
Run `bash scripts/run_all_minimal.sh` to execute the full suite for Colored MNIST.
