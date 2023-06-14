#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

MASKRCNN_HOME=$QUICKSTART_WORKSPACE/maskrcnn
MASKRCNN_MODEL=$MASKRCNN_HOME/model
MASKRCNN_DATA=$MASKRCNN_HOME/data
MASKRCNN_OUTPUT=$MASKRCNN_HOME/output

USE_IPEX=false
PRECISION=fp32

function usage(){
    echo "Usage: run-training.sh [ --ipex ] [ --precision fp32 | bf16 | bf32 ]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
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

export USE_IPEX

export DATASET_DIR=$MASKRCNN_DATA
export CHECKPOINT_DIR=$MASKRCNN_MODEL
export OUTPUT_DIR=$MASKRCNN_OUTPUT
mkdir -p $OUTPUT_DIR

cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/maskrcnn/training/cpu
bash training.sh $PRECISION
