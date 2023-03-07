#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
BERT_HOME=$ML_WORKSPACE/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data
BERT_OUTPUT=$BERT_HOME/output


# Clone the Transformers repo in the BERT large inference directory
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/bert_large/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.18.0
git apply ../enable_ipex_for_squad.diff
pip install -e ./

# Env vars
export FINETUNED_MODEL=$BERT_MODEL
export EVAL_DATA_FILE=$BERT_DATA
export OUTPUT_DIR=$BERT_OUTPUT

# Run a quickstart script (for example, FP32 multi-instance realtime inference)
bash run_multi_instance_realtime.sh fp32
