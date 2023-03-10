#!/bin/bash



RESNET50_HOME=$INTELAI_MODELS_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data
RESNET50_OUTPUT=$RESNET50_HOME/output

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


# Env vars
export DATASET_DIR=$RESNET50_DATA
export OUTPUT_DIR=$RESNET50_OUTPUT

#(fp32, int8, avx-int8, bf16, bf32 or fp16)
export PRECISION=$PRECISION
mkdir -p $OUTPUT_DIR
cd ${MODEL_DIR}/quickstart/image_recognition/pytorch/resnet50/inference/cpu

bash accuracy.sh
bash inference_throughput.sh
bash inference_realtime.sh
