#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

DISEASE_PREDICTION_DATA_OUTPUT=$DISEASE_PREDICTION_DATA/output
DISEASE_PREDICTION_DATA_TMP=$DISEASE_PREDICTION_DATA/tmp

RAW_DATA_PATH=""
PROCESSED_DATA_PATH=""

function usage(){
    echo "Usage: run-training.sh [ --processed-data-path ]"
    echo "Specify processed-data-path to the processed data."
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --processed-data-path)
        shift
        PROCESSED_DATA_PATH=$1
        ;;
    *)
        usage
    esac
    shift
done

if [ "${PROCESSED_DATA_PATH}" == "" ]; then
    # we assume user has run prepare-data
    PROCESSED_DATA_PATH=$DISEASE_PREDICTION_DATA/processed
fi

# create the directories
mkdir -p $DISEASE_PREDICTION_DATA_OUTPUT
mkdir -p $DISEASE_PREDICTION_DATA_TMP

echo "Output: $DISEASE_PREDICTION_DATA_OUTPUT"

python -u \
    $DISEASE_PREDICTION_HOME/src/train.py \
        --processed-data-path "${PROCESSED_DATA_PATH}" \
        --output-dir $DISEASE_PREDICTION_DATA_OUTPUT \
        --temp-dir $DISEASE_PREDICTION_DATA_TMP \
        "$@"
