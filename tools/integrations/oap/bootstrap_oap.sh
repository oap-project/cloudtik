#!/bin/bash

export USER_HOME=/home/$(whoami)
oap_install_dir=$USER_HOME/runtime/oap

if [ -n "$1" ];then
    oap_version=$1
else
    oap_version=1.4.0.spark32.h331
fi

# Install OAP by Conda
conda create -p "${oap_install_dir}"  -c conda-forge -c intel-beaver -c intel-bigdata -c intel -y oap=${oap_version}