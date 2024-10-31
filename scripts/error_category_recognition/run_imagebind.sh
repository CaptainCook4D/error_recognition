#!/bin/bash

declare -a SPLITS=('recordings' 'step' 'person' 'environment')
declare -a ERROR_CATEGORIES=('TechniqueError' 'PreparationError' 'MeasurementError' 'TemperatureError' 'TimingError')
declare -a VARIANTS=('Transformer')
declare -a BACKBONE=("imagebind")
declare -a MODALITY=('audio' 'video')
CKPT_DIRECTORY_PATH="/data/rohith/captain_cook/checkpoints/"
FEATURES_DIRECTORY="/data/rohith/captain_cook/features/gopro/segments_2"
TASK_NAME="error_category_recognition"

# Function name corrected for typo and best practice
generate_run_scripts() {
    local current_dir=$(pwd)  # Good practice to store the current directory if needed later
    for split in "${SPLITS[@]}"; do
        for modality in "${MODALITY[@]}"; do
            for error_category in "${ERROR_CATEGORIES[@]}"; do
                for variant in "${VARIANTS[@]}"; do
                    echo "Running the imagebind backbone for split: $split and variant: $variant and error category: $error_category and modality: $modality"
                    # Direct use of $BACKBONE since it's declared as a single-element array
                    if [[ "$variant" == "MLP" ]]; then
                        python train_ecr.py --error_category $error_category --task_name $TASK_NAME --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --modality $modality
                    elif [[ "$variant" == "Transformer" ]]; then
                        python train_ecr.py --error_category $error_category --task_name $TASK_NAME --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --lr 0.000001 --modality $modality
                    fi
                done
            done
        done
    done
}

# Corrected function call
generate_run_scripts
