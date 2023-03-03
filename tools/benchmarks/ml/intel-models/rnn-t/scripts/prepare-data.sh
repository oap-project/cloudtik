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

if [ ! -n "${MODEL_DIR}" ]; then
  echo "Please set environment variable '\${MODEL_DIR}'."
  exit 1
fi

function download_data_for_training() {
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh
}

function download_data_for_inference() {
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_dataset.sh
}

function download_model_for_inference() {
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_model.sh
}

download_data_for_training
download_data_for_inference
download_model_for_inference

