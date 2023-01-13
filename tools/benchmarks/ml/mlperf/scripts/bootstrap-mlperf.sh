#!/bin/bash

args=$(getopt -a -o h::p: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=/cloudtik/fs/benchmark-tools
MLPERF_HOME=$BENCHMARK_TOOL_HOME/mlperf
GITSPACE=$USER_HOME/gitspace

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

function install_tools() {
    #Installl cmake, GCC 9.0
    sudo apt-get install cmake -y
    sudo apt-get install gcc-9 g++-9 -y
}

function install_libaries() {

}

function install_mlperf() {
    pip install "git+https://github.com/mlcommons/logging.git@2.0.0"
}

function install_mlperf_scripts() {
  mkdir -p $GITSPACE
  cd $GITSPACE
  git clone https://github.com/oap-project/cloudtik.git
  rm -rf $MLPERF_HOME/*
  cp -r cloudtik/tools/benchmarks/ml/mlperf/* $MLPERF_HOME/


}

function configure_mlperf() {

}

prepare
install_tools
install_libaries
install_mlperf
install_mlperf_scripts
configure_mlperf
