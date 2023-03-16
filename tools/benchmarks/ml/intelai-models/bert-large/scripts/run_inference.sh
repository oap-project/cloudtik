#!/bin/bash



BERT_HOME=$INTELAI_MODELS_WORKSPACE/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data
BERT_OUTPUT=$BERT_HOME/output
SQUAD_DATA=$BERT_DATA/squad/dev-v1.1.json
SQUAD_MODEL=$BERT_MODEL/bert_squad_model

PRECISION=fp32
METRIC=throughput

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32 | bf16 | bf32] [--metric throughput | realtime] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --precision)
        # training or inference
        shift
        PRECISION=$1
        ;;
    --metric)
        shift
        METRIC=$1
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
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/bert_large/inference/cpu

# Run a quickstart script (for example, FP32 multi-instance realtime inference)

if [ "${METRIC}" = "throughput" ]; then
    bash run_multi_instance_throughput.sh $PRECISION
elif [ "${METRIC}" = "realtime" ]; then
    bash run_multi_instance_realtime.sh $PRECISION
else
    usage
fi