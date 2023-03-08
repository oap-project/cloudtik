#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
SSD_RESNET34_HOME=$ML_WORKSPACE/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data
SSD_RESNET34_OUTPUT=$SSD_RESNET34_HOME/output

export DATASET_DIR=$SSD_RESNET34_DATA
export CHECKPOINT_DIR=$SSD_RESNET34_MODEL
export OUTPUT_DIR=$SSD_RESNET34_OUTPUT
mkdir $OUTPUT_DIR

cd ${MODEL_DIR}/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
bash inference_throughput.sh fp32


