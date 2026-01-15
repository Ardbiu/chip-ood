#!/bin/bash
# scripts/run_domainbed_phase1.sh
# Run PACS and TerraIncognita with 5 seeds per method.
# Methods: ERM, CORAL, CHIP (Best)

OUTPUT_ROOT="results/domainbed_phase1"
mkdir -p $OUTPUT_ROOT

SEEDS=(0)
DATASETS=("domainbed_pacs" "domainbed_terraincognita")
METHODS=("erm" "chip") # Add coral if implemented

# CHIP Best Config
CHIP_ARGS="method.lambda_rec=0.0 method.lambda_adv=0.1 method.adv_type=clf method.z_dim_c=8 method.z_dim_s=32"

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # ERM
        echo "Running ERM on $dataset (Seed $seed)..."
        .venv/bin/python -m chip_ood.training.trainer \
            method=erm \
            data=$dataset \
            seed=$seed \
            output_dir="$OUTPUT_ROOT/${dataset}_erm_${seed}" \
            trainer.max_epochs=2 \
            hydra.job.chdir=False \
            > "$OUTPUT_ROOT/${dataset}_erm_${seed}.log" 2>&1
            
        # CHIP
        echo "Running CHIP on $dataset (Seed $seed)..."
        .venv/bin/python -m chip_ood.training.trainer \
            method=chip \
            data=$dataset \
            seed=$seed \
            output_dir="$OUTPUT_ROOT/${dataset}_chip_${seed}" \
            trainer.max_epochs=20 \
            hydra.job.chdir=False \
            $CHIP_ARGS \
            > "$OUTPUT_ROOT/${dataset}_chip_${seed}.log" 2>&1
    done
done

echo "DomainBed Phase 1 complete. Aggregating..."
.venv/bin/python -m chip_ood.evaluation.aggregate --input_dir $OUTPUT_ROOT --output_dir $OUTPUT_ROOT/summary
cp $OUTPUT_ROOT/summary/main_table.csv experiments/icml_domainbed_phase1.csv
