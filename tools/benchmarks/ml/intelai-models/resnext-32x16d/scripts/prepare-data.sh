#!/bin/bash

source ../../common/scripts/setenv.sh

ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
RESNEXT101_HOME=$ML_WORKSPACE/resnext-32x16d
RESNEXT101_DATA=$RESNEXT101_HOME/data

mkdir -p $RESNEXT101_DATA


function download_data() {
    mkdir -p $RESNEXT101_DATA
    cd $RESNEXT101_DATA
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
}

function prepare_val_data() {
    mkdir -p $RESNEXT101_DATA/val
    cd $RESNEXT101_DATA
    tar -xvf ILSVRC2012_img_val.tar -C $RESNEXT101_DATA/val

    cd $RESNEXT101_DATA/val
    wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    bash valprep.sh
}

download_data
prepare_val_data
