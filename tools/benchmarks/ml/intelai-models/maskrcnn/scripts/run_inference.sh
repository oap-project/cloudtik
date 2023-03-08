#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
MASKRCNN_HOME=$ML_WORKSPACE/maskrcnn
MASKRCNN_MODEL=$MASKRCNN_HOME/model
MASKRCNN_DATA=$MASKRCNN_HOME/data
MASKRCNN_OUTPUT=$MASKRCNN_HOME/output

export DATASET_DIR=$MASKRCNN_DATA
export CHECKPOINT_DIR=$MASKRCNN_MODEL
export OUTPUT_DIR=$MASKRCNN_OUTPUT
mkdir $OUTPUT_DIR

cd ${MODEL_DIR}/quickstart/object_detection/pytorch/maskrcnn/training/cpu
bash throughput.sh fp32
