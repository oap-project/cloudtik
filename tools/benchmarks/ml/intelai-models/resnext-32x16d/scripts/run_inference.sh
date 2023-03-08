#!/bin/bash

ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
RESNEXT101_HOME=$ML_WORKSPACE/resnext-32x16d
RESNEXT101_MODEL=$RESNEXT101_HOME/model
RESNEXT101_DATA=$RESNEXT101_HOME/data
RESNEXT101_OUTPUT=$RESNEXT101_HOME/output

function usage(){
    echo "Usage: run-inference.sh  [ --precision fp32| int8| avx-int8| bf16| bf32| fp16 ] "
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

# Env vars
export DATASET_DIR=$RESNEXT101_DATA
export OUTPUT_DIR=$RESNEXT101_OUTPUT
#(fp32, int8, avx-int8, bf16, bf32 or fp16)
export PRECISION=$PRECISION

cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu

bash accuracy.sh
bash inference_throughput.sh
bash inference_realtime.sh
