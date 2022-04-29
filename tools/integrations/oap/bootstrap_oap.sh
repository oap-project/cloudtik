#!/bin/bash

export USER_HOME=/home/$(whoami)
oap_install_dir=$USER_HOME/runtime/oap

# Install OAP by Conda
conda create -p "${oap_install_dir}" -c conda-forge -c intel -y oap=1.3.1
