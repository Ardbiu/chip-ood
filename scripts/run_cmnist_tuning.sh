#!/bin/bash
# scripts/run_cmnist_tuning.sh
# Sweep for stabilization (HSIC vs Clf, dimensions, lambdas)

OUTPUT_ROOT="results/colored_mnist_tuning"
mkdir -p $OUTPUT_ROOT

# Sweep Grid
SEEDS=(0 1 2) # 3 seeds for tuning
LAMBDA_ADVS=(0.1 1.0 10.0)
DIM_CONFIGS=("8_32" "16_64") # zc_zs

# Default: No Rec (since chip_adv_only was promising)
LAMBDA_REC=0.0

for seed in "${SEEDS[@]}"; do
    for l_adv in "${LAMBDA_ADVS[@]}"; do
        for dim_conf in "${DIM_CONFIGS[@]}"; do
             IFS='_' read -r zc zs <<< "$dim_conf"
             
             # 1. HSIC Variant
             NAME="hsic_la${l_adv}_zc${zc}_zs${zs}_${seed}"
             echo "Running $NAME..."
             .venv/bin/python -m chip_ood.training.trainer \
                method=chip \
                data=colored_mnist \
                seed=$seed \
                output_dir="$OUTPUT_ROOT/$NAME" \
                trainer.max_epochs=10 \
                ++data.variant="binary_irm" \
                method.lambda_rec=$LAMBDA_REC \
                method.lambda_adv=$l_adv \
                method.adv_type="hsic" \
                method.z_dim_c=$zc \
                method.z_dim_s=$zs \
                method.name="chip_hsic" \
                > "$OUTPUT_ROOT/$NAME.log" 2>&1 &
                
             # 2. CLF Variant (Adversarial) - Retrying with different lambdas
             NAME="clf_la${l_adv}_zc${zc}_zs${zs}_${seed}"
             echo "Running $NAME..."
             .venv/bin/python -m chip_ood.training.trainer \
                method=chip \
                data=colored_mnist \
                seed=$seed \
                output_dir="$OUTPUT_ROOT/$NAME" \
                trainer.max_epochs=10 \
                ++data.variant="binary_irm" \
                method.lambda_rec=$LAMBDA_REC \
                method.lambda_adv=$l_adv \
                method.adv_type="clf" \
                method.z_dim_c=$zc \
                method.z_dim_s=$zs \
                method.name="chip_clf" \
                > "$OUTPUT_ROOT/$NAME.log" 2>&1 &
        done
        wait # Batch size control (wait for seed-batch)
    done
done

echo "Tuning sweep complete. Aggregating..."
.venv/bin/python -m chip_ood.evaluation.aggregate --input_dir $OUTPUT_ROOT --output_dir $OUTPUT_ROOT/summary
