#!/bin/bash

ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
RNNT_HOME=$ML_WORKSPACE/rnn-t
RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

mkdir -p $DATASET_DIR
mkdir -p $CHECKPOINT_DIR


PHASE="inference"

if [ ! -n "${MODEL_DIR}" ]; then
  echo "Please set environment variable '\${MODEL_DIR}'."
  exit 1
fi

function usage(){
    echo "Usage: prepare-data.sh  [ --phase training | inference] "
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
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu/download_dataset.sh
elif [ "${PHASE}" != "inference" ]; then
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_dataset.sh
    bash ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_model.sh
else
    usage
fi

