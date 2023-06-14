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
mkdir -p $CHECKPOINT_DIR

NUM_STEPS=100
USE_IPEX=false
#(fp32, bf16, bf32)
PRECISION=fp32

function usage(){
    echo "Usage: run-training.sh [ --num-steps 100 ] [ --ipex ] [ --precision fp32 | bf16 | bf32 ] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --num-steps)
        # num for steps
        shift
        NUM_STEPS=$1
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
export NUM_STEPS

cd ${QUICKSTART_HOME}/scripts/language_modeling/pytorch/rnnt/training/cpu

bash training.sh $PRECISION
