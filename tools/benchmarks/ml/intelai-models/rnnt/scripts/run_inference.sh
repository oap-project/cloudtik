#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

RNNT_HOME=$INTELAI_MODELS_WORKSPACE/rnn-t

RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

# Env vars
export OUTPUT_DIR=$RNNT_OUTPUT_DIR
export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

mkdir -p $OUTPUT_DIR

#(fp32, bf16, bf32)
PRECISION=fp32
USE_IPEX=0

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32 | bf16 | bf32] [--use-ipex]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --precision)
        shift
        PRECISION=$1
        ;;
    --use-ipex)
        USE_IPEX=1
        ;;
    *)
        usage
    esac
    shift
done

export USE_IPEX

cd ${SCRIPT_DIR}

bash accuracy.sh $PRECISION
bash inference_throughput.sh $PRECISION
bash inference_realtime.sh $PRECISION
