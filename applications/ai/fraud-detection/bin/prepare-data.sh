#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

FRAUD_DETECTION_WORKING_DATA=$FRAUD_DETECTION_WORKING/data
RAW_DATA_PATH=""
RAW_DATA_ARCHIVE=""

function usage(){
    echo "Usage: prepare-data.sh [ --raw-data-path ] [ --raw-data-archive ]"
    echo "Specify either raw-data-path to the raw data file or directory or "
    echo "raw-data-archive to the tgz file contains the raw data."
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --raw-data-path)
        shift
        RAW_DATA_PATH=$1
        ;;
    --raw-data-archive)
        shift
        RAW_DATA_ARCHIVE=$1
        ;;
    *)
        usage
    esac
    shift
done

function download_data() {
    mkdir -p $FRAUD_DETECTION_WORKING_DATA
    cd $FRAUD_DETECTION_WORKING_DATA
    # TODO: download TabFormer dataset if possible
    # download from https://github.com/IBM/TabFormer/tree/main/data/credit_card/transactions.tgz
}

function prepare_data() {
    if [ "${RAW_DATA_PATH}" == "" ]; then
        mkdir -p $FRAUD_DETECTION_WORKING_DATA/raw
        cd $FRAUD_DETECTION_WORKING_DATA
        tar -zxvf ${RAW_DATA_ARCHIVE} -C $FRAUD_DETECTION_WORKING_DATA/raw
        RAW_DATA_PATH=$FRAUD_DETECTION_WORKING_DATA/raw
    fi

    PROCESSED_DATA_PATH=$FRAUD_DETECTION_WORKING_DATA/processed/processed_data.csv
    cloudtik head run ai.modeling.xgboost --single-node --no-train \
        --raw-data-path ${RAW_DATA_PATH} \
        --processed-data-path ${PROCESSED_DATA_PATH} \
        "$@"
}

if [ "${RAW_DATA_PATH}" == "" ] && [ "${RAW_DATA_ARCHIVE}" == "" ]; then
    usage
fi

prepare_data
move_to_workspace $FRAUD_DETECTION_WORKING_DATA
