#!/bin/bash

#declare -a SPLITS=('recordings' 'person' 'environment' 'step')
declare -a SPLITS=('person' 'environment')
declare -a VARIANTS=('MLP' 'Transformer')
declare -a BACKBONE=("imagebind")
#declare -a MODALITY=('audio' 'video' 'video audio' 'video audio text' 'audio text' 'video text')
declare -a MODALITY=('audio' 'video')
CKPT_DIRECTORY_PATH="/data/rohith/captain_cook/checkpoints/"
FEATURES_DIRECTORY="/data/rohith/captain_cook/features/gopro/segments"
TASK_NAME=("early_error_recognition")

generate_run_scripts() {
    local current_dir=$(pwd)
    for modality in "${MODALITY[@]}"; do
        for split in "${SPLITS[@]}"; do
            for variant in "${VARIANTS[@]}"; do
                echo "Running the imagebind backbone for split: $split and variant: $variant and modality: $modality"
                if [[ "$variant" == "MLP" ]]; then
                    python train_er.py --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --modality $modality
                elif [[ "$variant" == "Transformer" ]]; then
                    python train_er.py --split $split --variant $variant --backbone ${BACKBONE[0]} --ckpt_directory $CKPT_DIRECTORY_PATH --lr 0.0001 --modality $modality
                fi
            done
        done
    done
}

# Corrected function call
generate_run_scripts
