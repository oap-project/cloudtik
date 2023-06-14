#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

SSD_RESNET34_HOME=$QUICKSTART_WORKING/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data

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


function download_inference_data() {
    mkdir -p $SSD_RESNET34_DATA
    export DATASET_DIR=$SSD_RESNET34_DATA

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/ssd-resnet34/inference/cpu
    bash download_dataset.sh
}


function prepare_inference_model() {
    mkdir -p $SSD_RESNET34_MODEL
    export CHECKPOINT_DIR=$SSD_RESNET34_MODEL

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/ssd-resnet34/inference/cpu
    bash download_model.sh
}


function download_training_data() {
    mkdir -p $SSD_RESNET34_DATA
    export DATASET_DIR=$SSD_RESNET34_DATA

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/ssd-resnet34/training/cpu
    bash download_dataset.sh
}


function prepare_training_model() {
    mkdir -p $SSD_RESNET34_MODEL
    export CHECKPOINT_DIR=$SSD_RESNET34_MODEL

    cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/ssd-resnet34/training/cpu
    bash download_model.sh

}


if [ "${PHASE}" = "training" ]; then
    download_training_data
    prepare_training_model
elif [ "${PHASE}" = "inference" ]; then
    download_inference_data
    prepare_inference_model
else
    usage
fi

move_to_workspace $SSD_RESNET34_HOME
