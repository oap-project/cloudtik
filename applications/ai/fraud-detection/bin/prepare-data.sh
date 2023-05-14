#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

FRAUD_DETECTION_WORKING_DATA=$FRAUD_DETECTION_WORKING/data
DATA_ARCHIVE_FILE=""

if [ ! -n "${FRAUD_DETECTION_WORKSPACE}" ]; then
  echo "Please set environment variable '\${FRAUD_DETECTION_WORKSPACE}'."
  exit 1
fi

function usage(){
    echo "Usage: prepare-data.sh  [ --data_archive_file the path to transactions.tgz file] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --data_archive_file)
        # training or inference
        shift
        DATA_ARCHIVE_FILE=$1
        ;;
    *)
        usage
    esac
    shift
done

function download_data() {
    mkdir -p $FRAUD_DETECTION_WORKING_DATA
    cd $FRAUD_DETECTION_WORKING_DATA
    # TODO
    # download from https://github.com/IBM/TabFormer/tree/main/data/credit_card/transactions.tgz
}

function prepare_data() {
    mkdir -p $FRAUD_DETECTION_WORKING_DATA/raw
    cd $FRAUD_DETECTION_WORKING_DATA
    tar -zxvf ${DATA_ARCHIVE_FILE} -C $FRAUD_DETECTION_WORKING_DATA/raw

    python $FRAUD_DETECTION_HOME/python/gnn/process_data.py \
        --raw_data_file $FRAUD_DETECTION_WORKING_DATA/raw/transactions.csv \
        --output_file $FRAUD_DETECTION_WORKING_DATA/processed/transactions.csv
}

if [ "${DATA_ARCHIVE_FILE}" = "" ]; then
    usage
fi

prepare_data
move_to_workspace $FRAUD_DETECTION_WORKING_DATA
