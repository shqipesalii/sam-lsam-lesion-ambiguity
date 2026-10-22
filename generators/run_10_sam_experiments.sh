#!/bin/bash

mkdir -p results

SEEDS=(42 123 2024 2025 7 77 99 1001 1337 9999)

for i in {1..10}
do
    SEED=${SEEDS[$i-1]}

    echo "======================================"
    echo "Running SAM experiment $i with seed $SEED"
    echo "======================================"

    DATASET_DIR="PhantomShapes_V2_run${i}"
    SAM_DIR="SAM_dataset_run${i}"
    PRED_DIR="SAM_predictions_run${i}"
    RESULT_CSV="results/results_run${i}.csv"

    python generate_v2.py \
        --seed $SEED \
        --out_dir $DATASET_DIR

    python prepare_sam_dataset.py \
        --input_dir $DATASET_DIR \
        --out_dir $SAM_DIR

    python run_sam_inference.py \
        --image_dir $SAM_DIR/images \
        --mask_dir $SAM_DIR/masks \
        --pred_dir $PRED_DIR \
        --checkpoint checkpoints/sam_vit_b_01ec64.pth \
        --model_type vit_b

    python evaluate_sam_full_metrics.py \
        --gt_dir $SAM_DIR/masks \
        --pred_dir $PRED_DIR \
        --metadata_csv $SAM_DIR/slice_metadata.csv \
        --output_csv $RESULT_CSV \
        --run_id run${i}

done

python aggregate_sam_runs.py \
    --results_dir results \
    --output_csv sam_10_runs_all_results.csv \
    --summary_csv sam_10_runs_summary.csv

python plot_sam_results.py \
    --input_csv sam_10_runs_all_results.csv

