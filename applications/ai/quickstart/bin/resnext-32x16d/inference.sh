#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

RESNEXT101_HOME=$QUICKSTART_WORKSPACE/resnext-32x16d

RESNEXT101_DATA=$RESNEXT101_HOME/data
RESNEXT101_OUTPUT=$RESNEXT101_HOME/output

mkdir -p $RESNEXT101_OUTPUT

USE_IPEX=false
#(fp32, int8, avx-int8, bf16, bf32 or fp16)
PRECISION=fp32

function usage(){
    echo "Usage: inference.sh [ --ipex ] [ --precision fp32| int8| avx-int8| bf16| bf32| fp16 ] "
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

# Env vars
export DATASET_DIR=$RESNEXT101_DATA
export OUTPUT_DIR=$RESNEXT101_OUTPUT

export USE_IPEX
export PRECISION

cd ${QUICKSTART_HOME}/scripts/image_recognition/pytorch/resnext-32x16d/inference/cpu

bash accuracy.sh
bash inference_throughput.sh
bash inference_realtime.sh
