#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../../common/scripts/setenv.sh

MASKRCNN_HOME=$INTELAI_MODELS_WORKSPACE/maskrcnn
MASKRCNN_MODEL=$MASKRCNN_HOME/model
MASKRCNN_DATA=$MASKRCNN_HOME/data
MASKRCNN_OUTPUT=$MASKRCNN_HOME/output

PRECISION=fp32
MODE=jit
METRIC=throughput
USE_IPEX=0

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32 | bf16 | bf32 ] [ --mode jit or imperative]  [--metric throughput | realtime] [ --ipex]"
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
    --mode)
        shift
        MODE=$1
        ;;
    --metric)
        shift
        METRIC=$1
        ;;
    --use-ipex)
        USE_IPEX=1
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
#<set to 'jit' or 'imperative'>
set MODE=$MODE

mkdir -p $OUTPUT_DIR

cd ${MODELS_HOME}/quickstart/object_detection/pytorch/maskrcnn/inference/cpu

if [ "${METRIC}" = "throughput" ]; then
    bash inference_throughput.sh $PRECISION $MODE
elif [ "${METRIC}" = "realtime" ]; then
    bash inference_realtime.sh $PRECISION $MODE
else
    usage
fi
