#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
SSD_RESNET34_HOME=$ML_WORKSPACE/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data

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

function install_libraries() {
    pip install --no-cache-dir cython
    pip install matplotlib Pillow pycocotools defusedxml
    pip install --no-cache-dir pytz==2018.5
}

function download_inference_data() {
    mkdir -p $SSD_RESNET34_DATA
    export DATASET_DIR=$SSD_RESNET34_DATA

    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
    bash download_dataset.sh
}


function prepare_inference_model() {
    mkdir -p $SSD_RESNET34_MODEL
    export CHECKPOINT_DIR=$SSD_RESNET34_MODEL

    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
    bash download_model.sh
}


function download_training_data() {
    mkdir -p $SSD_RESNET34_DATA
    export DATASET_DIR=$SSD_RESNET34_DATA

    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
    bash download_dataset.sh
}


function prepare_training_model() {
    mkdir -p $SSD_RESNET34_MODEL
    export CHECKPOINT_DIR=$SSD_RESNET34_MODEL

    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
    bash download_model.sh

}


install_libraries
if [ "${PHASE}" = "training" ]; then
    download_training_data
    prepare_training_model
elif [ "${PHASE}" != "inference" ]; then
    download_inference_data
    prepare_inference_model
else
    usage
fi
