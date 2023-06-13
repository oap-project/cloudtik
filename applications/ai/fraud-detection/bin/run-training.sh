#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

FRAUD_DETECTION_DATA_OUTPUT=$FRAUD_DETECTION_DATA/output
FRAUD_DETECTION_DATA_TMP=$FRAUD_DETECTION_DATA/tmp
mkdir -p $FRAUD_DETECTION_DATA_OUTPUT
mkdir -p $FRAUD_DETECTION_DATA_TMP

TABULAR2GRAPH=${SCRIPT_DIR}/../config/tabular2graph.yaml

python -u $FRAUD_DETECTION_HOME/bin/train.py \
        --raw-data-path $FRAUD_DETECTION_WORKSPACE/raw \
        --output-dir $FRAUD_DETECTION_DATA_OUTPUT \
        --temp-dir $FRAUD_DETECTION_DATA_TMP \
        --tabular2graph $TABULAR2GRAPH \
        "$@"
