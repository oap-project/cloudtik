#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../../common/scripts/setenv.sh

RESNET50_HOME=$INTELAI_MODELS_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data
RESNET50_OUTPUT=$RESNET50_HOME/output

PRECISION=fp32
METRIC=throughput

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32 | bf16 | bf32] [--metric throughput | realtime]"
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
    --metric)
        shift
        METRIC=$1
        ;;
    *)
        usage
    esac
    shift
done


# Env vars
export DATASET_DIR=$RESNET50_DATA
export OUTPUT_DIR=$RESNET50_OUTPUT

#(fp32, int8, avx-int8, bf16, bf32 or fp16)
export PRECISION=$PRECISION
mkdir -p $OUTPUT_DIR
cd ${MODELS_HOME}/quickstart/image_recognition/pytorch/resnet50/inference/cpu

if [ "${METRIC}" = "throughput" ]; then
    bash inference_throughput.sh
elif [ "${METRIC}" = "realtime" ]; then
    bash inference_realtime.sh
else
    usage
fi
