#!/bin/bash

args=$(getopt -a -o h::p: -l head:: -- "$@")
eval set -- "${args}"

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
    export USER_HOME=/home/$(whoami)
    BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
    MLPERF_HOME=$BENCHMARK_TOOL_HOME/mlperf
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

}

function configure_mlperf() {

}

prepare
install_tools
install_libaries
install_mlperf
configure_mlperf
