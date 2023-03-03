#!/bin/bash

USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
RNNT_HOME=$ML_WORKSPACE/rnn-t
RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

OUTPUT_DIR=RNNT_OUTPUT_DIR
DATASET_DIR=RNNT_DATASET_DIR
CHECKPOINT_DIR=RNNT_CHECKPOINT_DIR


function run_training() {
    cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu
    bash training.sh fp32
}

function run_inference() {
    cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu
    bash inference_realtime.sh fp32
}
