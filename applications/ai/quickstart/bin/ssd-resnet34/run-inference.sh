#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

SSD_RESNET34_HOME=$QUICKSTART_WORKSPACE/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data
SSD_RESNET34_OUTPUT=$SSD_RESNET34_HOME/output

METRIC=throughput
USE_IPEX=false
PRECISION=fp32

function usage(){
    echo "Usage: run-inference.sh [ --metric throughput | realtime ] [ --ipex ] [ --num-steps 100 ] [ --precision fp32 | bf16 | bf32 ]"
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

export USE_IPEX

export DATASET_DIR=$SSD_RESNET34_DATA
export CHECKPOINT_DIR=$SSD_RESNET34_MODEL
export OUTPUT_DIR=$SSD_RESNET34_OUTPUT
mkdir -p $OUTPUT_DIR

cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/ssd-resnet34/inference/cpu

if [ "${METRIC}" = "throughput" ]; then
    bash inference_throughput.sh $PRECISION
elif [ "${METRIC}" = "realtime" ]; then
    bash inference_realtime.sh $PRECISION
else
    usage
fi
