#!/bin/bash

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
export OAP_HOME=$RUNTIME_PATH/oap

if [ -n "$1" ];then
    OAP_VERSION=$1
else
    OAP_VERSION=1.4.0.spark32
fi

# Install OAP by Conda
if [ ! -d "${OAP_HOME}" ]; then
    conda create -p "${OAP_HOME}" -c conda-forge -c intel-beaver -c intel-bigdata -c intel -y oap=${OAP_VERSION}
fi
