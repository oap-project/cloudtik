#!/bin/bash

USER_HOME=/home/$(whoami)
APPLICATIONS_HOME=$USER_HOME/applications
QUICKSTART_TMP=/tmp/quickstart

# Application path on cluster node
export QUICKSTART_HOME=${APPLICATIONS_HOME}/quickstart
export MODEL_DIR=$QUICKSTART_HOME

# Working path on the local machine
if test -e "/mnt/cloudtik/data_disk_1/"
then
    QUICKSTART_WORKING=/mnt/cloudtik/data_disk_1/quickstart
else
    QUICKSTART_WORKING=$USER_HOME/quickstart
fi

# Workspace path on shared storage
if test -e "/cloudtik/fs"
then
    QUICKSTART_WORKSPACE="/cloudtik/fs/quickstart"
else
    QUICKSTART_WORKSPACE=$QUICKSTART_WORKING
fi

function prepare() {
    source ~/.bashrc
    sudo apt-get update -y

    mkdir -p $QUICKSTART_HOME
    sudo chown $(whoami) $QUICKSTART_HOME

    mkdir -p $QUICKSTART_WORKING
    sudo chown $(whoami) $QUICKSTART_WORKING

    mkdir -p $QUICKSTART_WORKSPACE
    sudo chown $(whoami) $QUICKSTART_WORKSPACE
}

function install_tools() {
    sudo apt-get update -y
    sudo apt-get install curl unzip -y
    sudo apt-get install gcc g++ cmake -y
    sudo apt-get install autoconf -y
}

function install_libaries() {
    pip -qq install gdown
    pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip
    pip install onnx==1.12.0
    pip install lark-parser hypothesis
    CLOUDTIK_CONDA_ENV=$(dirname $(dirname $(which cloudtik)))
    conda install ninja dataclasses -p $CLOUDTIK_CONDA_ENV -y
}

function install_quickstart() {
    mkdir -p $QUICKSTART_TMP
    cd $QUICKSTART_TMP
    rm -rf $QUICKSTART_TMP/cloudtik
    git clone https://github.com/oap-project/cloudtik.git
    rm -rf $QUICKSTART_HOME/*
    mkdir -p $QUICKSTART_HOME
    cp -r cloudtik/applications/ai/quickstart/* $QUICKSTART_HOME/
    rm -rf $QUICKSTART_TMP/cloudtik
}

function install_models_dependency() {
    for dir in $QUICKSTART_HOME/bin/*/; do
        if [ -d ${dir} ]; then
            install_dependency_script_path="${dir}/install-dependency.sh"
            if [ -e $install_dependency_script_path ]; then
                bash $install_dependency_script_path
            fi
        fi
    done
}

prepare
install_tools
install_libaries
install_quickstart
install_models_dependency
