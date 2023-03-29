#!/bin/bash

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools

# Tool path on local machine
INTELAI_MODELS_HOME=$BENCHMARK_TOOL_HOME/intelai_models
MODELS_HOME=$INTELAI_MODELS_HOME/models
SCRIPTS_HOME=$INTELAI_MODELS_HOME/scripts
MODELS_TMP=$INTELAI_MODELS_HOME/tmp

# Working path on the local machine
if test -e "/mnt/cloudtik/data_disk_1/"
then
    INTELAI_MODELS_WORKING=/mnt/cloudtik/data_disk_1/intelai_models
else
    INTELAI_MODELS_WORKING=$USER_HOME/intelai_models
fi

# Workspace path on shared storage
if test -e "/cloudtik/fs"
then
    INTELAI_MODELS_WORKSPACE="/cloudtik/fs/intelai_models"
else
    INTELAI_MODELS_WORKSPACE=$INTELAI_MODELS_WORKING
fi

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

function is_head_node() {
    if [ -n $IS_HEAD_NODE ]; then
        cloudtik head head-ip
        GET_HEAD_IP_CODE=$?
        if [ ${GET_HEAD_IP_CODE} -eq "0" ]; then
            IS_HEAD_NODE=true
        else
            IS_HEAD_NODE=false
        fi
    fi
}

function prepare() {
    source ~/.bashrc
    sudo apt-get update -y

    mkdir -p $INTELAI_MODELS_HOME
    sudo chown $(whoami) $INTELAI_MODELS_HOME

    mkdir -p $INTELAI_MODELS_WORKING
    sudo chown $(whoami) $INTELAI_MODELS_WORKING

    mkdir -p $INTELAI_MODELS_WORKSPACE
    sudo chown $(whoami) $INTELAI_MODELS_WORKSPACE
}

function install_intelai_models() {
  cd $INTELAI_MODELS_HOME
  rm -rf models
  git clone https://github.com/IntelAI/models.git
}

function install_tools() {
    sudo apt-get update -y
    sudo apt-get install curl unzip -y
    sudo apt-get install numactl gcc g++ cmake -y
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

function install_intelai_models_scripts() {
    mkdir -p $MODELS_TMP
    cd $MODELS_TMP
    rm -rf $MODELS_TMP/cloudtik
    git clone https://github.com/oap-project/cloudtik.git
    rm -rf SCRIPTS_HOME/*
    mkdir -p $SCRIPTS_HOME
    cp -r cloudtik/tools/benchmarks/ml/intelai-models/* $SCRIPTS_HOME/
    rm -rf $MODELS_TMP/cloudtik
}

function configure_intelai_models() {
    # Nothing to do now
    :
}


function install_models_dependency() {
    for dir in $SCRIPTS_HOME/*/; do
        if [ -d ${dir} ]; then
            install_dependency_script_path="${dir}scripts/install-dependency.sh"
            if [ -e $install_dependency_script_path ]; then
                bash install_dependency_script_path
            fi
        fi
    done
}


prepare
install_tools
install_libaries
install_intelai_models
install_intelai_models_scripts
install_models_dependency
configure_intelai_models
