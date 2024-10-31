#!/bin/bash

declare -a SPLITS=('step')
declare -a VARIANTS=('Transformer')
declare -a BACKBONE=("imagebind")
declare -a MODALITY=('video audio text depth')
CKPT_DIRECTORY_PATH="/data/rohith/captain_cook/checkpoints/"
FEATURES_DIRECTORY="/data/rohith/captain_cook/features/gopro/segments_2"
TASK_NAME="error_recognition"

# Function name corrected for typo and best practice
generate_run_scripts() {
    local current_dir=$(pwd)  # Good practice to store the current directory if needed later
    for split in "${SPLITS[@]}"; do
        for modality in "${MODALITY[@]}"; do
            for variant in "${VARIANTS[@]}"; do
                echo "Running the imagebind multimodal backbone for split: $split and variant: $variant and error category: $error_category and modality: $modality"
                # Direct use of $BACKBONE since it's declared as a single-element array
                if [[ "$variant" == "MLP" ]]; then
                    python train_er.py --task_name $TASK_NAME --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --modality $modality
                elif [[ "$variant" == "Transformer" ]]; then
                    python train_er.py --task_name $TASK_NAME --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --lr 0.000001 --modality $modality
                fi
            done
        done
    done
}

# Corrected function call
generate_run_scripts
