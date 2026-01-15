#!/bin/bash
# scripts/run_cmnist_icml.sh
# Runs ICML-grade experiments on ColoredMNIST
# 10 Seeds, 4 Variants of CHIP, plus ERM.

SEEDS=({0..4}) # Start with 5 seeds for speed, user can extend to 9
OUTPUT_ROOT="results/colored_mnist_icml"

mkdir -p $OUTPUT_ROOT

run_exp() {
    METHOD=$1
    SEED=$2
    ARGS=$3
    NAME="${METHOD}_${SEED}"
    
    echo "Running $METHOD (Seed $SEED)..."
    .venv/bin/python -m chip_ood.training.trainer \
        method=$METHOD \
        data=colored_mnist \
        seed=$SEED \
        output_dir="$OUTPUT_ROOT/$NAME" \
        trainer.max_epochs=10 \
        ++data.variant="binary_irm" \
        $ARGS \
        > "$OUTPUT_ROOT/$NAME.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
    # 1. ERM
    run_exp "erm" "$seed" ""
    
    # 2. CHIP (Full)
    run_exp "chip" "$seed" "method.lambda_rec=1.0 method.lambda_adv=1.0"
    
    # 3. CHIP (No Adv) - Ablation
    run_exp "chip" "$seed" "method.lambda_rec=1.0 method.lambda_adv=0.0 method.name=chip_no_adv"
    
    # 4. CHIP (No Rec) - Ablation
    run_exp "chip" "$seed" "method.lambda_rec=0.0 method.lambda_adv=1.0 method.name=chip_no_rec"
    
    # 5. CHIP (Adv Only / Constraint) - "Minimal"
    # Strong bottleneck on Zc, no decoder
    run_exp "chip" "$seed" "method.lambda_rec=0.0 method.lambda_adv=1.0 method.z_dim_c=8 method.name=chip_adv_only"
done

echo "Experiments complete. Aggregating..."
.venv/bin/python -m chip_ood.evaluation.aggregate --input_dir $OUTPUT_ROOT --output_dir $OUTPUT_ROOT/summary
