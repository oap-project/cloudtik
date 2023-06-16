#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

BERT_HOME=$QUICKSTART_WORKSPACE/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data
BERT_OUTPUT=$BERT_HOME/output
SQUAD_DATA=$BERT_DATA/squad/dev-v1.1.json
SQUAD_MODEL=$BERT_MODEL/bert_squad_model

METRIC=throughput
PRECISION=fp32
USE_IPEX=false

function usage(){
    echo "Usage: inference.sh [ --metric throughput | realtime ] [ --ipex ] [ --precision fp32 | bf16 | bf32 ] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --metric)
        shift
        METRIC=$1
        ;;
    --ipex)
        shift
        USE_IPEX=true
        ;;
    --precision)
        shift
        PRECISION=$1
        ;;
    *)
        usage
    esac
    shift
done


# Env vars
export FINETUNED_MODEL=$SQUAD_MODEL
export EVAL_DATA_FILE=$SQUAD_DATA
export OUTPUT_DIR=$BERT_OUTPUT
mkdir -p $OUTPUT_DIR

export USE_IPEX
export PRECISION

cd ${QUICKSTART_HOME}/scripts/language_modeling/pytorch/bert_large/inference/cpu

# Run a quickstart script (for example, FP32 multi-instance realtime inference)

if [ "${METRIC}" = "throughput" ]; then
    bash run_multi_instance_throughput.sh $PRECISION
elif [ "${METRIC}" = "realtime" ]; then
    bash run_multi_instance_realtime.sh $PRECISION
else
    usage
fi
