#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

MASKRCNN_HOME=$INTELAI_MODELS_WORKING/maskrcnn
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

install_tools
install_libraries