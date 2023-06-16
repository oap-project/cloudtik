#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

DISEASE_PREDICTION_WORKING_DATA=$DISEASE_PREDICTION_WORKING/data

function usage(){
    echo "Usage: prepare-data.sh [ --no-download ]"
    exit 1
}

NO_DOWNLOAD=NO

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --no-download)
        NO_DOWNLOAD=YES
        ;;
    *)
        usage
    esac
    shift
done

function download_data() {
    mkdir -p $DISEASE_PREDICTION_WORKING_DATA/raw
    python -u \
      $SCRIPT_DIR/src/data/process.py \
        --no-process \
        --data-path $DISEASE_PREDICTION_WORKING_DATA/raw
}

function prepare_data() {
    if [ ! $NO_DOWNLOAD ]; then
        download_data
    if

    mkdir -p $DISEASE_PREDICTION_WORKING_DATA/processed
    python -u \
      $SCRIPT_DIR/src/data/process.py \
        --no-download \
        --data-path $DISEASE_PREDICTION_WORKING_DATA/raw \
        --output-dir $DISEASE_PREDICTION_WORKING_DATA/processed
}

prepare_data
move_to_workspace $DISEASE_PREDICTION_WORKING_DATA
