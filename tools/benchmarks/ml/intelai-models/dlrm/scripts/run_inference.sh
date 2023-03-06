#!/bin/bash

USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
DLRM_HOME=$ML_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data
DLRM_OUTPUT=$BERT_HOME/output

export PRECISION=bf16
export DATASET_DIR=$DLRM_DATA
export OUTPUT_DIR=$DLRM_OUTPUT

# Run a quickstart script (for example, bare metal performance)
cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/inference/cpu
bash inference_performance.sh
