#!/bin/bash


USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
BERT_HOME=$ML_WORKSPACE/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data
BERT_OUTPUT=$BERT_HOME/output

export OUTPUT_DIR=$BERT_OUTPUT
export DATASET_DIR=$BERT_DATA
export TRAIN_SCRIPT=${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/run_pretrain_mlperf.py



# For phase 1 get the bert config from <https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT>

export BERT_MODEL_CONFIG=$BERT_MODEL/bert_config.json

# Run the phase 1 quickstart script for fp32 (or bf16)
cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/bert_large/training/cpu

# Remove dense_seq_output option
line_number=`grep -n "dense_seq_output" run_bert_pretrain_phase1.sh | cut -d: -f1`
sed -i -e $line_number"d" run_bert_pretrain_phase1.sh

bash run_bert_pretrain_phase1.sh bf16

