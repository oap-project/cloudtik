#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
MASKRCNN_HOME=$ML_WORKSPACE/maskrcnn
MASKRCNN_MODEL=$MASKRCNN_HOME/model
MASKRCNN_DATA=$MASKRCNN_HOME/data


function install_tools() {
     sudo apt install build-essential -y
     sudo apt-get install libgl1 -y

}


function install_libraries() {
    pip install yacs opencv-python pycocotools defusedxml cityscapesscripts
    conda install -n cloudtik intel-openmp -y

}

function download_data() {
    
    mkdir -p $MASKRCNN_DATA
    export DATASET_DIR=$MASKRCNN_DATA

    cd $MODEL_DIR/quickstart/object_detection/pytorch/maskrcnn/inference/cpu
    bash download_dataset.sh
    cd $MODEL_DIR/quickstart/object_detection/pytorch/maskrcnn/training/cpu
    bash download_dataset.sh
}


function prepare_model() {
    export CHECKPOINT_DIR=$RESNET34_MODEL
    # Install model
    cd models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/
    python setup.py develop

    cd $MODEL_DIR/quickstart/object_detection/pytorch/maskrcnn/inference/cpu
    bash download_model.sh

}


install_libraries
download_data
prepare_model
