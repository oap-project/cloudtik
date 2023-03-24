#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

RESNET50_HOME=$INTELAI_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data

PHASE="inference"

if [ ! -n "${MODEL_DIR}" ]; then
  echo "Please set environment variable '\${MODEL_DIR}'."
  exit 1
fi

function usage(){
    echo "Usage: prepare-data.sh  [ --phase training | inference] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --phase)
        # training or inference
        shift
        PHASE=$1
        ;;
    *)
        usage
    esac
    shift
done

function download_inference_data() {
    mkdir -p $RESNET50_DATA/val
    cd $RESNET50_DATA
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
}

function download_training_data() {
    
    mkdir -p $RESNET50_DATA
    cd $RESNET50_DATA
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
#    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar
#    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

}


function prepare_training_data() {
    mkdir -p $RESNET50_DATA/train
    cd $RESNET50_DATA
    tar -xvf ILSVRC2012_img_train.tar -C $RESNET50_DATA/train

    for tar_file in $RESNET50_DATA/train/*.tar; do
        mkdir -p $RESNET50_DATA/train/$(basename $tar_file .tar);
        tar -xvf $tar_file -C /home/cloudtik/mltest/resnet50/train/$(basename $tar_file .tar);
    done
    cd $RESNET50_DATA/train
    rm -rf ./*.tar
}


function prepare_val_data() {
    mkdir -p $RESNET50_DATA/val
    cd $RESNET50_DATA
    tar -xvf ILSVRC2012_img_val.tar -C $RESNET50_DATA/val

    cd $RESNET50_DATA/val
    wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    bash valprep.sh
}

if [ "${PHASE}" = "training" ]; then
    download_training_data
    download_inference_data
    prepare_training_data
    prepare_val_data
elif [ "${PHASE}" = "inference" ]; then
    download_inference_data
    prepare_val_data
else
    usage
fi
move_to_shared_dict $RESNET50_HOME
