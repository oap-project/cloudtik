#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

RESNET50_HOME=$QUICKSTART_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data
RESNET50_OUTPUT=$RESNET50_HOME/output

METRIC=throughput
USE_IPEX=false
PRECISION=fp32

function usage(){
    echo "Usage: inference.sh [ --metric throughput | realtime ] [ --ipex ] [ --precision fp32 | bf16 | bf32 ]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --metric)
        shift
        METRIC=$1
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

export USE_IPEX
#(fp32, int8, avx-int8, bf16, bf32 or fp16)
export PRECISION

cd ${QUICKSTART_HOME}/scripts/image_recognition/pytorch/resnet50/inference/cpu

if [ "${METRIC}" = "throughput" ]; then
    bash inference_throughput.sh
elif [ "${METRIC}" = "realtime" ]; then
    bash inference_realtime.sh
else
    usage
fi
