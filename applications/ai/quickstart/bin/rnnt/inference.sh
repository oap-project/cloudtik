#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

RNNT_HOME=$QUICKSTART_WORKSPACE/rnn-t

RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

# Env vars
export OUTPUT_DIR=$RNNT_OUTPUT_DIR
export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

mkdir -p $OUTPUT_DIR

USE_IPEX=false
#(fp32, bf16, bf32)
PRECISION=fp32

function usage(){
    echo "Usage: inference.sh [ --ipex ] [ --precision fp32 | bf16 | bf32 ]"
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

cd ${QUICKSTART_HOME}/scripts/language_modeling/pytorch/rnnt/inference/cpu

bash accuracy.sh $PRECISION
bash inference_throughput.sh $PRECISION
bash inference_realtime.sh $PRECISION
