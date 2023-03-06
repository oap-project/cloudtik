#!/bin/bash

USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
RNNT_HOME=$ML_WORKSPACE/rnn-t
RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

# Env vars
export OUTPUT_DIR=RNNT_OUTPUT_DIR
export DATASET_DIR=RNNT_DATASET_DIR
export CHECKPOINT_DIR=RNNT_CHECKPOINT_DIR

#(fp32, bf16, bf32)
PRECISION=fp32

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32 | bf16 | bf32] "
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
    *)
        usage
    esac
    shift
done

cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu

bash accuracy.sh $PRECISION
bash inference_throughput.sh $PRECISION
bash inference_realtime.sh $PRECISION
