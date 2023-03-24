#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

MASKRCNN_HOME=$INTELAI_MODELS_LOCAL_WORKSPACE/maskrcnn
MASKRCNN_MODEL=$MASKRCNN_HOME/model
MASKRCNN_DATA=$MASKRCNN_HOME/data


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


function install_tools() {
     sudo apt install build-essential -y
     sudo apt-get install libgl1 -y

}


function install_libraries() {
    pip install yacs opencv-python pycocotools defusedxml cityscapesscripts
    conda install -n cloudtik intel-openmp -y

}

function download_training_data() {
    mkdir -p $MASKRCNN_DATA
    export DATASET_DIR=$MASKRCNN_DATA

    cd $MODEL_DIR/quickstart/object_detection/pytorch/maskrcnn/training/cpu
    bash download_dataset.sh
}


function prepare_training_model() {
    export CHECKPOINT_DIR=$MASKRCNN_MODEL
    # Install model
    cd $MODEL_DIR/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/
    python setup.py develop
}


function download_inference_data() {
    mkdir -p $MASKRCNN_DATA
    export DATASET_DIR=$MASKRCNN_DATA

    cd $MODEL_DIR/quickstart/object_detection/pytorch/maskrcnn/inference/cpu
    bash download_dataset.sh
}


function prepare_inference_model() {
    export CHECKPOINT_DIR=$MASKRCNN_MODEL
    # Install model
    cd $MODEL_DIR/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/
    python setup.py develop

    cd $MODEL_DIR/quickstart/object_detection/pytorch/maskrcnn/inference/cpu
    bash download_model.sh
}

install_tools
install_libraries

if [ "${PHASE}" = "training" ]; then
    download_training_data
    prepare_training_model
elif [ "${PHASE}" = "inference" ]; then
    download_inference_data
    prepare_inference_model
else
    usage
fi

move_to_shared_dict $MASKRCNN_HOME
