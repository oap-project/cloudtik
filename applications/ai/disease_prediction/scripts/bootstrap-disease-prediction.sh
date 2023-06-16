#!/bin/bash

USER_HOME=/home/$(whoami)
APPLICATIONS_HOME=$USER_HOME/applications

# Application path on local machine
DISEASE_PREDICTION_HOME=${APPLICATIONS_HOME}/disease_prediction
DISEASE_PREDICTION_TMP=/tmp/disease_prediction

# Working path on the local machine
if test -e "/mnt/cloudtik/data_disk_1/"
then
    DISEASE_PREDICTION_WORKING=/mnt/cloudtik/data_disk_1/disease_prediction
else
    DISEASE_PREDICTION_WORKING=$USER_HOME/disease_prediction
fi

# Workspace path on shared storage, working path if shared storage not exists
if test -e "/cloudtik/fs"
then
    DISEASE_PREDICTION_WORKSPACE="/cloudtik/fs/disease_prediction"
else
    DISEASE_PREDICTION_WORKSPACE=$DISEASE_PREDICTION_WORKING
fi

function prepare() {
    source ~/.bashrc
    sudo apt-get update -y

    mkdir -p $DISEASE_PREDICTION_HOME
    sudo chown $(whoami) $DISEASE_PREDICTION_HOME

    mkdir -p $DISEASE_PREDICTION_WORKING
    sudo chown $(whoami) $DISEASE_PREDICTION_WORKING

    mkdir -p $DISEASE_PREDICTION_WORKSPACE
    sudo chown $(whoami) $DISEASE_PREDICTION_WORKSPACE
}

function install_tools() {
    :;
}

function install_libaries() {
    pip install --no-cache-dir -qq docx2txt openpyxl pillow
}

function install_disease_prediction() {
    mkdir -p $DISEASE_PREDICTION_TMP
    cd $DISEASE_PREDICTION_TMP
    rm -rf $DISEASE_PREDICTION_TMP/cloudtik
    git clone https://github.com/oap-project/cloudtik.git
    rm -rf $DISEASE_PREDICTION_HOME/*
    mkdir -p $DISEASE_PREDICTION_HOME
    cp -r cloudtik/applications/ai/disease_prediction/* $DISEASE_PREDICTION_HOME/
    rm -rf $DISEASE_PREDICTION_TMP/cloudtik
}

prepare
install_tools
install_libaries
install_disease_prediction
