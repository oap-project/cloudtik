#!/bin/bash

ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
DLRM_HOME=$ML_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data
DLRM_OUTPUT=$BERT_HOME/output

export PRECISION=bf16
export DATASET_DIR=$DLRM_DATA
export OUTPUT_DIR=$DLRM_OUTPUT

cd ${MODEL_DIR}/quickstart/recommendation/pytorch/dlrm/training/cpu

NUM_BATCH=10000 bash training.sh
