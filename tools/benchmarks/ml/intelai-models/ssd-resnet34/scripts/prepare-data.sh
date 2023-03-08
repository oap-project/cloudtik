#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
SSD_RESNET34_HOME=$ML_WORKSPACE/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data


function install_libraries() {
    pip install --no-cache-dir cython
    pip install matplotlib Pillow pycocotools defusedxml
    pip install --no-cache-dir pytz==2018.5
}

function download_data() {
    
    mkdir -p $SSD_RESNET34_DATA
    export DATASET_DIR=$SSD_RESNET34_DATA

    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
    bash download_dataset.sh
    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
    bash download_dataset.sh
}


function prepare_model() {
    export CHECKPOINT_DIR=$SSD_RESNET34_MODEL

    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
    bash download_model.sh
    cd $MODEL_DIR/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
    bash download_model.sh

}



install_libraries
download_data
prepare_model
