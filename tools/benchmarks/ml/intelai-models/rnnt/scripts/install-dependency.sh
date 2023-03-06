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

PHASE="inference"

if [ ! -n "${MODEL_DIR}" ]; then
  echo "Please set environment variable '\${MODEL_DIR}'."
  exit 1
fi


function usage(){
    echo "Usage: install-dependency.sh  [ --phase training | inference] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --phase)
        # training or inference
        shift
        PHASE=$1
        ;;
    *)
        usage
    esac
    shift
done

if [ "${PHASE}" = "training" ]; then
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu/install_dependency.sh
elif [ "${PHASE}" != "inference" ]; then
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/install_dependency_baremetal.sh
else
    usage
fi
