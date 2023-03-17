#!/bin/bash

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
MODELS_HOME=$BENCHMARK_TOOL_HOME/models
MODELS_SCRIPTS_HOME=$BENCHMARK_TOOL_HOME/intelai_models_scripts
MODELS_TMP=$USER_HOME/intelai_models/tmp

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
    mkdir -p $BENCHMARK_TOOL_HOME
    sudo chown $(whoami) $BENCHMARK_TOOL_HOME
}

function install_intelai_models() {
  cd $BENCHMARK_TOOL_HOME
  rm -rf models
  git clone https://github.com/IntelAI/models.git

  if test -e "/mnt/cloudtik/data_disk_1/"
  then
      INTELAI_MODELS_WORKSPACE=/mnt/cloudtik/data_disk_1/intelai_models_workspace
  else
      INTELAI_MODELS_WORKSPACE=$USER_HOME/intelai_models_workspace
  fi
  mkdir -p $INTELAI_MODELS_WORKSPACE

}

function install_tools() {
    sudo apt-get install curl unzip -y
    sudo apt-get install numactl gcc g++ cmake -y
    sudo apt-get install autoconf -y
}

function install_libaries() {
    pip -qq install gdown
    pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip
    pip install sklearn onnx
    pip install lark-parser hypothesis
    CLOUDTIK_CONDA_ENV=$(dirname $(dirname $(which cloudtik)))
    conda install numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses psutil -p $CLOUDTIK_CONDA_ENV
}

function install_intelai_models_scripts() {
    mkdir -p $MODELS_TMP
    cd $MODELS_TMP
    rm -rf $MODELS_TMP/*
    git clone https://github.com/oap-project/cloudtik.git
    rm -rf MODELS_SCRIPTS_HOME/*
    mkdir -p $MODELS_SCRIPTS_HOME
    cp -r cloudtik/tools/benchmarks/ml/intelai-models/* $MODELS_SCRIPTS_HOME/
    rm -rf $MODELS_TMP/cloudtik
}

function configure_intelai_models() {
    # Nothing to do now
    :
}

prepare
install_tools
install_libaries
install_intelai_models
install_intelai_models_scripts
configure_intelai_models
