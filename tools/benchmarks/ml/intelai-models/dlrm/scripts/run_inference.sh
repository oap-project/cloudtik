#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

DLRM_HOME=$INTELAI_MODELS_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data
DLRM_OUTPUT=$BERT_HOME/output

PRECISION=fp32

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32 | bf16 | bf32] "
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
    *)
        usage
    esac
    shift
done

export PRECISION=$PRECISION
export DATASET_DIR=$DLRM_DATA
export OUTPUT_DIR=$DLRM_OUTPUT
mkdir -p $OUTPUT_DIR
# Run a quickstart script (for example, bare metal performance)
cd ${MODELS_HOME}/quickstart/recommendation/pytorch/dlrm/inference/cpu
bash inference_performance.sh
