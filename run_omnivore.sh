#!/bin/bash

declare -a SPLITS=('recordings' 'person' 'environment' 'step')
declare -a VARIANTS=('MLP' 'Transformer')
declare -a BACKBONE=("omnivore")
CKPT_DIRECTORY_PATH="/data/rohith/captain_cook/checkpoints/"
FEATURES_DIRECTORY="/data/rohith/captain_cook/features/gopro/segments"

# Function name corrected for typo and best practice
generate_run_scripts() {
    local current_dir=$(pwd)  # Good practice to store the current directory if needed later
    for split in "${SPLITS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            echo "Running the omnivore backbone for split: $split and variant: $variant"
            # Direct use of $BACKBONE since it's declared as a single-element array
            if [[ "$variant" == "MLP" ]]; then
                python train_er.py --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH
            elif [[ "$variant" == "Transformer" ]]; then
                python train_er.py --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --lr 0.0001
            fi
        done
    done
}

# Corrected function call
generate_run_scripts
