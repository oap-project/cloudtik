#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
BERT_HOME=$ML_WORKSPACE/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data
BERT_OUTPUT=$BERT_HOME/output
SQUAD_DATA=$BERT_DATA/squad
SQUAD_MODEL=$BERT_MODEL/bert_squad_model

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
        # training or inference
        shift
        PRECISION=$1
        ;;
    *)
        usage
    esac
    shift
done


# Clone the Transformers repo in the BERT large inference directory
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/bert_large/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.18.0
git apply ../enable_ipex_for_squad.diff
pip install -e ./

# Env vars
export FINETUNED_MODEL=$SQUAD_MODEL
export EVAL_DATA_FILE=$SQUAD_DATA
export OUTPUT_DIR=$BERT_OUTPUT
mkdir -p $OUTPUT_DIR

# Run a quickstart script (for example, FP32 multi-instance realtime inference)
bash run_multi_instance_realtime.sh $PRECISION
