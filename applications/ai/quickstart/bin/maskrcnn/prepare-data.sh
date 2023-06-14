#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

MASKRCNN_HOME=$QUICKSTART_WORKING/maskrcnn
MASKRCNN_MODEL=$MASKRCNN_HOME/model
MASKRCNN_DATA=$MASKRCNN_HOME/data


PHASE="inference"

if [ ! -n "${QUICKSTART_HOME}" ]; then
  echo "Please set environment variable '\${QUICKSTART_HOME}'."
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




function download_training_data() {
    mkdir -p $MASKRCNN_DATA
    export DATASET_DIR=$MASKRCNN_DATA

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/maskrcnn/training/cpu
    bash download_dataset.sh
}



function download_inference_data() {
    mkdir -p $MASKRCNN_DATA
    export DATASET_DIR=$MASKRCNN_DATA

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/maskrcnn/inference/cpu
    bash download_dataset.sh
}


function download_inference_model() {
    export CHECKPOINT_DIR=$MASKRCNN_MODEL

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/maskrcnn/inference/cpu
    bash download_model.sh
}


if [ "${PHASE}" = "training" ]; then
    download_training_data
elif [ "${PHASE}" = "inference" ]; then
    download_inference_data
    download_inference_model
else
    usage
fi

move_to_workspace $MASKRCNN_HOME
