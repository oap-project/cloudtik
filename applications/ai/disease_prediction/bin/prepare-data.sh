#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

DISEASE_PREDICTION_WORKING_DATA=$DISEASE_PREDICTION_WORKING/data

function usage(){
    echo "Usage: prepare-data.sh [ --image-path ] [ --no-download ] "
    exit 1
}

NO_DOWNLOAD=false

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --image-path)
        shift
        IMAGE_PATH=$1
        ;;
    --no-download)
        NO_DOWNLOAD=true
        ;;
    *)
        usage
    esac
    shift
done

function download_data() {
    RAW_DATA_PATH=$DISEASE_PREDICTION_WORKING_DATA/raw
    mkdir -p $RAW_DATA_PATH

    export PYTHONPATH=$DISEASE_PREDICTION_HOME/src:${PYTHONPATH}
    python -u \
      $DISEASE_PREDICTION_HOME/src/process.py \
        --no-process \
        --no-split \
        --dataset-path $RAW_DATA_PATH
}

function prepare_data() {
    if [ "$NO_DOWNLOAD" != "true" ]; then
        download_data
    fi

    PROCESSED_DATA_PATH=$DISEASE_PREDICTION_WORKING_DATA/processed
    mkdir -p $PROCESSED_DATA_PATH

    export PYTHONPATH=$DISEASE_PREDICTION_HOME/src:${PYTHONPATH}
    python -u \
      $DISEASE_PREDICTION_HOME/src/process.py \
        --no-download \
        --dataset-path $DISEASE_PREDICTION_WORKING_DATA/raw \
        --image-path $IMAGE_PATH \
        --output-dir $PROCESSED_DATA_PATH
}
if [ "${IMAGE_PATH}" == "" ]; then
    usage
fi

prepare_data
move_to_workspace $DISEASE_PREDICTION_WORKING_DATA
