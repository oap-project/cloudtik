#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

RNNT_HOME=$INTELAI_MODELS_WORKING/rnn-t

RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

export OUTPUT_DIR=$RNNT_OUTPUT_DIR
export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

PHASE="inference"

if [ ! -n "${MODELS_DIR}" ]; then
  echo "Please set environment variable '\${MODELS_DIR}'."
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
    bash ${MODELS_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu/install_dependency.sh
elif [ "${PHASE}" = "inference" ]; then
    bash ${MODELS_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu/install_dependency_baremetal.sh
else
    usage
fi
