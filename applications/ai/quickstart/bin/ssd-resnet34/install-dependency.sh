#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

SSD_RESNET34_HOME=$QUICKSTART_WORKING/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data


function install_libraries() {
    pip install --no-cache-dir cython
    pip install matplotlib Pillow pycocotools defusedxml
    pip install --no-cache-dir pytz
}

install_libraries
