#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

RESNET50_HOME=$QUICKSTART_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data
RESNET50_OUTPUT=$RESNET50_HOME/output

TRAINING_EPOCHS=5
USE_IPEX=false
PRECISION=fp32

function usage(){
    echo "Usage: train.sh [ --training_epochs ] [ --ipex ] [ --precision fp32 | bf16 | bf32 ]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --training_epochs)
        # num for steps
        shift
        TRAINING_EPOCHS=$1
        ;;
    --ipex)
        shift
        USE_IPEX=true
        ;;
    --precision)
        shift
        PRECISION=$1
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

export TRAINING_EPOCHS=$TRAINING_EPOCHS
export USE_IPEX
export PRECISION

# Run the training quickstart script
cd ${QUICKSTART_HOME}/scripts/image_recognition/pytorch/resnet50/training/cpu
bash training.sh
