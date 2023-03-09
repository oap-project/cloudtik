#!/bin/bash


ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
SSD_RESNET34_HOME=$ML_WORKSPACE/ssd-resnet34
SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
SSD_RESNET34_DATA=$SSD_RESNET34_HOME/data
SSD_RESNET34_OUTPUT=$SSD_RESNET34_HOME/output

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

export DATASET_DIR=$SSD_RESNET34_DATA
export CHECKPOINT_DIR=$SSD_RESNET34_MODEL
export OUTPUT_DIR=$SSD_RESNET34_OUTPUT
mkdir -p $OUTPUT_DIR

cd ${MODEL_DIR}/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu
bash throughput.sh $PRECISION
