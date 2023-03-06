#!/bin/bash


USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
RESNET50_HOME=$ML_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data
RESNET50_OUTPUT=$RESNET50_HOME/output

# Env vars
export DATASET_DIR=$RESNET50_DATA
export OUTPUT_DIR=$RESNET50_OUTPUT

#(fp32, int8, avx-int8, bf16, bf32 or fp16)
export PRECISION=fp32

cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/resnet50/inference/cpu

bash accuracy.sh
bash inference_throughput.sh
bash inference_realtime.sh