#!/bin/bash


USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
SSD-RESNET34_HOME=$ML_WORKSPACE/ssd-resnet34
SSD-RESNET34_MODEL=$SSD-RESNET34_HOME/model
SSD-RESNET34_DATA=$SSD-RESNET34_HOME/data
SSD-RESNET34_OUTPUT=$SSD-RESNET34_HOME/output

export DATASET_DIR=$SSD-RESNET34_DATA
export CHECKPOINT_DIR=$SSD-RESNET34_MODEL
export OUTPUT_DIR=$SSD-RESNET34_OUTPUT

cd ${MODEL_DIR}/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
bash throughput.sh fp32
