#!/bin/bash
# scripts/run_cmnist_10seed.sh
# Final verification of the BEST configuration from tuning.

OUTPUT_ROOT="results/colored_mnist_10seed"
mkdir -p $OUTPUT_ROOT

# Best Config (Placeholder - will update after reading table)
# method=chip_clf lambda_adv=10.0 zc=8 zs=32 (Hypothetically)
METHOD="chip_clf"
ARGS="method.lambda_rec=0.0 method.lambda_adv=0.1 method.adv_type=clf method.z_dim_c=8 method.z_dim_s=32 ++data.variant=binary_irm"

SEEDS=({0..9})

for seed in "${SEEDS[@]}"; do
    NAME="${METHOD}_${seed}"
    echo "Running $NAME..."
    .venv/bin/python -m chip_ood.training.trainer \
        method=chip \
        data=colored_mnist \
        seed=$seed \
        output_dir="$OUTPUT_ROOT/$NAME" \
        trainer.max_epochs=15 \
        method.name="$METHOD" \
        $ARGS \
        > "$OUTPUT_ROOT/$NAME.log" 2>&1
done

echo "10-seed run complete. Aggregating..."
.venv/bin/python -m chip_ood.evaluation.aggregate --input_dir $OUTPUT_ROOT --output_dir $OUTPUT_ROOT/summary
cp $OUTPUT_ROOT/summary/main_table.csv experiments/icml_cmnist_10seed.csv
