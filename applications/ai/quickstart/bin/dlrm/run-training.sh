#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

DLRM_HOME=$QUICKSTART_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data
DLRM_OUTPUT=$BERT_HOME/output

NUM_BATCH=10000
USE_IPEX=false
PRECISION=fp32


function usage(){
    echo "Usage: run-training.sh [ --num-batch ] [ --ipex ] [ --precision fp32 | bf16 | bf32 ]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --num-batch)
        # num for steps
        shift
        NUM_BATCH=$1
        ;;
    --ipex)
        shift
        USE_IPEX=true
        ;;
    --precision)
        # training or inference
        shift
        PRECISION=$1
        ;;
    *)
        usage
    esac
    shift
done

export USE_IPEX
export PRECISION

export DATASET_DIR=$DLRM_DATA
export OUTPUT_DIR=$DLRM_OUTPUT
mkdir -p $OUTPUT_DIR
cd ${QUICKSTART_HOME}/scripts/recommendation/pytorch/dlrm/training/cpu

NUM_BATCH=$NUM_BATCH bash training.sh
