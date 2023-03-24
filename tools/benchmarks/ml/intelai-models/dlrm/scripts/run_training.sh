#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

DLRM_HOME=$INTELAI_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data
DLRM_OUTPUT=$BERT_HOME/output

PRECISION=fp32
NUM_BATCH=10000

function usage(){
    echo "Usage: run-training.sh  [--precision fp32 | bf16 | bf32]  [--num-batch]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --precision)
        # training or inference
        shift
        PRECISION=$1
        ;;
    --num-batch)
        # num for steps
        shift
        NUM_BATCH=$1
        ;;
    *)
        usage
    esac
    shift
done

export PRECISION=$PRECISION
export DATASET_DIR=$DLRM_DATA
export OUTPUT_DIR=$DLRM_OUTPUT
mkdir -p $OUTPUT_DIR
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/training/cpu

NUM_BATCH=$NUM_BATCH bash training.sh
