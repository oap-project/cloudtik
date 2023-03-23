#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

RESNET50_HOME=$INTELAI_MODELS_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data
RESNET50_OUTPUT=$RESNET50_HOME/output

PRECISION=fp32
TRAINING_EPOCHS=5
function usage(){
    echo "Usage: run-training.sh  [--precision fp32 | bf16 | bf32]  [--training_epochs]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --precision)
        shift
        PRECISION=$1
        ;;
    --training_epochs)
        # num for steps
        shift
        TRAINING_EPOCHS=$1
        ;;
    *)
        usage
    esac
    shift
done

# Env vars
export DATASET_DIR=$RESNET50_DATA
export OUTPUT_DIR=$RESNET50_OUTPUT
mkdir -p $OUTPUT_DIR

export PRECISION=$PRECISION
export TRAINING_EPOCHS=$TRAINING_EPOCHS

# Run the training quickstart script
cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/resnet50/training/cpu
bash training.sh
