#!/bin/bash
set -e

PROJECT_DIR="/home/shqipe/Desktop/sam_phantom_project"
NNUNET_DIR="$PROJECT_DIR/nnunet_2d_project"

export nnUNet_raw="$NNUNET_DIR/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_DIR/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_DIR/nnUNet_results"

for RUN in {1..10}
do
    INPUT_DIR="$PROJECT_DIR/SAM_10_runs/PhantomShapes_V2_run${RUN}/imagesTr"
    OUTPUT_DIR="$NNUNET_DIR/predictions/nnunet2d_run${RUN}"

    mkdir -p "$OUTPUT_DIR"

    nnUNetv2_predict \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        -d 502 \
        -c 2d \
        -f 0 \
        -tr nnUNetTrainer_100epochs
done

echo "All 10 predictions finished."
