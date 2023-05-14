#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/configure.sh

FRAUD_DETECTION_DATA_OUTPUT=$FRAUD_DETECTION_DATA/output
FRAUD_DETECTION_DATA_TMP=$FRAUD_DETECTION_DATA/tmp
mkdir -p $FRAUD_DETECTION_DATA_OUTPUT
mkdir -p $FRAUD_DETECTION_DATA_TMP

TABULAR2GRAPH=${SCRIPT_DIR}/../config/tabular2graph.yaml

python $FRAUD_DETECTION_HOME/python/gnn/run.py \
        --input_file $FRAUD_DETECTION_DATA/processed/transactions.csv \
        --output_dir $FRAUD_DETECTION_DATA_OUTPUT \
        --tmp_dir $FRAUD_DETECTION_DATA_TMP \
        --tabular2graph $TABULAR2GRAPH \
        "$@"
