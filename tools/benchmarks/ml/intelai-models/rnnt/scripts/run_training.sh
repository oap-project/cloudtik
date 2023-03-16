#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

RNNT_HOME=$INTELAI_MODELS_WORKSPACE/rnn-t

RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

# Env vars
export OUTPUT_DIR=$RNNT_OUTPUT_DIR
export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

#(fp32, bf16, bf32)
PRECISION=fp32
NUM_STEPS=100

function usage(){
    echo "Usage: run-training.sh  [--precision fp32 | bf16 | bf32]  [--num-steps]"
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
    --num-steps)
        # num for steps
        shift
        NUM_STEPS=$1
        ;;
    *)
        usage
    esac
    shift
done

cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu
NUM_STEPS=${NUM_STEPS} bash training.sh bash training.sh $PRECISION
