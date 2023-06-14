#!/bin/bash

USER_HOME=/home/$(whoami)

function prepare() {
    source ~/.bashrc
}

function install_ipex() {
    pip -qq install intel-extension-for-pytorch==1.13.0
    CLOUDTIK_CONDA_ENV=$(dirname $(dirname $(which cloudtik)))
    conda install intel-openmp jemalloc mkl mkl-include -p $CLOUDTIK_CONDA_ENV -y
}

prepare
install_ipex
