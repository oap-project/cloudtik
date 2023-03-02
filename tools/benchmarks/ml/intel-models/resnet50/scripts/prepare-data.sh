#!/bin/bash


USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
RESNET50_HOME=$ML_WORKSPACE/resnet50
RESNET50_MODEL=$RESNET50_HOME/model
RESNET50_DATA=$RESNET50_HOME/data

function download_data() {
    
    mkdir -p $RESNET50_DATA
    cd $RESNET50_DATA
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

}


function prepare_train_data() {
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

download_data
prepare_train_data
prepare_val_data
