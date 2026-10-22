#!/bin/bash
set -e

PROJECT_DIR="/home/shqipe/Desktop/sam_phantom_project"
NNUNET_DIR="$PROJECT_DIR/nnunet_2d_project"

export nnUNet_raw="$NNUNET_DIR/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_DIR/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_DIR/nnUNet_results"

DATASET_ID="502"
DATASET_NAME="Dataset502_PhantomShapesV2_2D"

mkdir -p "$nnUNet_raw/$DATASET_NAME"

cp -r "$PROJECT_DIR/SAM_10_runs/LSAM_train_PhantomShapes_V2/imagesTr" \
"$nnUNet_raw/$DATASET_NAME/"

cp -r "$PROJECT_DIR/SAM_10_runs/LSAM_train_PhantomShapes_V2/labelsTr" \
"$nnUNet_raw/$DATASET_NAME/"

cp "$PROJECT_DIR/SAM_10_runs/LSAM_train_PhantomShapes_V2/dataset.json" \
"$nnUNet_raw/$DATASET_NAME/"

echo "Dataset copied to: $nnUNet_raw/$DATASET_NAME"
echo "Now run:"
echo "nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity"
