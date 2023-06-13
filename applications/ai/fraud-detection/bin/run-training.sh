#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

FRAUD_DETECTION_DATA_OUTPUT=$FRAUD_DETECTION_DATA/output
FRAUD_DETECTION_DATA_TMP=$FRAUD_DETECTION_DATA/tmp
TABULAR2GRAPH=${SCRIPT_DIR}/../config/tabular2graph.yaml

RAW_DATA_PATH=""
PROCESSED_DATA_PATH=""

function usage(){
    echo "Usage: run-training.sh [ --raw-data-path ] [ --processed-data-path ]"
    echo "Specify either raw-data-path to the raw data file or directory or"
    echo "processed-data-path to the processed data file."
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
    --processed-data-path)
        shift
        PROCESSED_DATA_PATH=$1
        ;;
    *)
        usage
    esac
    shift
done

if [ "${RAW_DATA_PATH}" == "" ] && [ "${PROCESSED_DATA_PATH}" == "" ]; then
    # we assume user has run prepare-data
    PROCESSED_DATA_PATH=$FRAUD_DETECTION_DATA/processed/processed_data.csv
fi

# create the directories
mkdir -p $FRAUD_DETECTION_DATA_OUTPUT
mkdir -p $FRAUD_DETECTION_DATA_TMP

echo "Output: $FRAUD_DETECTION_DATA_OUTPUT"

python -u $FRAUD_DETECTION_HOME/bin/train.py \
        --raw-data-path "${RAW_DATA_PATH}" \
        --processed-data-path "${PROCESSED_DATA_PATH}" \
        --output-dir $FRAUD_DETECTION_DATA_OUTPUT \
        --temp-dir $FRAUD_DETECTION_DATA_TMP \
        --tabular2graph $TABULAR2GRAPH \
        "$@"
